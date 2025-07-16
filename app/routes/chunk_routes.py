import traceback
from typing import List, Dict, Optional
from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel
import numpy as np
from app.routes.document_routes import get_cached_query_embedding
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import hashlib
from langchain_core.runnables import run_in_executor

from app.config import logger, vector_store
from app.services.vector_store.async_pg_vector import AsyncPgVector

class ChunkExtractionRequest(BaseModel):
    documentIds: List[str]
    maxChunks: int = 50
    minChunkLength: int = 100
    maxChunkLength: int = 1000
    diversityWeight: float = 0.7  # Balance between relevance and diversity

class ChunkResponse(BaseModel):
    chunkId: str
    content: str
    metadata: Dict
    score: float
    characteristics: Dict  # Why this chunk was selected

router = APIRouter()

@router.get("/document-chunks", response_model=List[ChunkResponse])
async def get_document_chunks(
    request: Request,
    documentIds: List[str] = Query(..., description="List of document IDs"),
    maxChunks: int = Query(50, description="Maximum number of chunks to return"),
    minChunkLength: int = Query(100, description="Minimum chunk length"),
    maxChunkLength: int = Query(1000, description="Maximum chunk length"),
    diversityWeight: float = Query(0.7, description="Balance between relevance and diversity"),
    query: Optional[str] = Query(None, description="Optional query to filter chunks")
):
    """
    Extract the best document chunks using a hybrid strategy:
    1. Information density (good for questions)
    2. Diversity (covers different topics)
    3. Completeness (self-contained chunks)
    4. Question indicators (chunks that naturally prompt questions)
    
    These chunks can be used by other services for FAQ generation.
    """
    
    try:
        # 1. Get all chunks for the specified documents
        all_chunks = await get_all_chunks_for_documents(document_ids=documentIds, query=query)
        
        if not all_chunks:
            raise HTTPException(status_code=404, detail="No chunks found for the specified files")

        # 2. Deduplicate chunks FIRST
        unique_chunks = deduplicate_chunks(all_chunks)
        
        # 3. Filter chunks by length
        filtered_chunks = [
            chunk for chunk in unique_chunks
            if minChunkLength <= len(chunk["content"]) <= maxChunkLength
        ]
        
        if not filtered_chunks:
            filtered_chunks = unique_chunks
        
        # 4. Score chunks for FAQ suitability
        scored_chunks = score_chunks_for_faq(filtered_chunks)
        
        # 5. Select diverse, high-quality chunks
        selected_chunks = select_diverse_chunks(
            scored_chunks,
            max_chunks=maxChunks,
            diversity_weight=diversityWeight
        )
        
        # 6. Format response
        return [
            ChunkResponse(
                chunkId=chunk["id"],
                content=chunk["content"],
                metadata=chunk["metadata"],
                score=chunk["faq_score"],
                characteristics=chunk["characteristics"]
            )
            for chunk in selected_chunks
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def get_all_chunks_for_documents(document_ids: List[str], query: Optional[str] = None) -> List[Dict]:
    """
    Retrieve all chunks for given document_ids with metadata
    """
    chunks = []
    
    try:
        # For AsyncPgVector, we should use its methods properly
        if isinstance(vector_store, AsyncPgVector):
            # Query based only on file_id
            filter_dict = {
                "file_id": {"$in": document_ids}
            }
            
            # Use the provided query if available, otherwise use a default query that doesn't filter
            search_query = query if query else ""  # Empty string for no specific filtering
            
            all_docs = await run_in_executor(
                None,
                lambda: vector_store.similarity_search_with_score(
                    query=search_query,
                    k=10000,      # High number to get all
                    filter=filter_dict
                )
            )
            
            # Process results
            seen_contents = set()  # Track unique content
            for idx, (doc, score) in enumerate(all_docs):
                file_id = doc.metadata.get('file_id')
                
                # Skip duplicates based on content
                content_hash = get_content_hash(doc.page_content)
                if content_hash in seen_contents:
                    continue
                seen_contents.add(content_hash)
                
                if file_id in document_ids:
                    chunks.append({
                        "id": f"{file_id}_{idx}",
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "embedding": None
                    })
                    
        else:
            # For other vector stores
            filter_dict = {
                "file_id": {"$in": document_ids}
            }
            
            # Use the provided query if available, otherwise use a default query that doesn't filter
            search_query = query if query else ""
            
            all_docs = vector_store.similarity_search_with_score(
                query=search_query,
                k=10000,
                filter=filter_dict
            )
            
            # Process results with deduplication
            seen_contents = set()
            for idx, (doc, score) in enumerate(all_docs):
                content_hash = get_content_hash(doc.page_content)
                if content_hash in seen_contents:
                    continue
                seen_contents.add(content_hash)
                
                file_id = doc.metadata.get('file_id')
                if file_id in document_ids:
                    chunks.append({
                        "id": f"{file_id}_{idx}",
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "embedding": None
                    })
                    
    except Exception as e:
        raise
    
    return chunks


def get_content_hash(content: str) -> str:
    """Generate a hash of normalized content for deduplication"""
    # Normalize whitespace and case for better matching
    normalized = " ".join(content.lower().split())
    return hashlib.md5(normalized.encode()).hexdigest()


def deduplicate_chunks(chunks: List[Dict], similarity_threshold: float = 0.95) -> List[Dict]:
    """
    Remove duplicate or near-duplicate chunks based on content similarity
    """
    if len(chunks) <= 1:
        return chunks
    
    # Use TF-IDF to find similar chunks
    texts = [chunk["content"] for chunk in chunks]
    
    try:
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Keep track of which chunks to keep
        keep_indices = []
        seen_groups = []
        
        for i in range(len(chunks)):
            # Check if this chunk is similar to any we've already kept
            is_duplicate = False
            
            for group in seen_groups:
                # Calculate similarity with first member of each group
                similarity = cosine_similarity(
                    tfidf_matrix[i:i+1], 
                    tfidf_matrix[group[0]:group[0]+1]
                )[0][0]
                
                if similarity >= similarity_threshold:
                    is_duplicate = True
                    group.append(i)  # Add to existing group
                    break
            
            if not is_duplicate:
                keep_indices.append(i)
                seen_groups.append([i])  # Start new group
        
        # Return only unique chunks, preferring higher scores
        unique_chunks = []
        for group in seen_groups:
            # From each group, pick the chunk with highest FAQ score if available
            # Otherwise, pick the first one
            if all('faq_score' in chunks[idx] for idx in group):
                best_idx = max(group, key=lambda idx: chunks[idx].get("faq_score", 0))
            else:
                best_idx = group[0]
            unique_chunks.append(chunks[best_idx])
        
        return unique_chunks
        
    except Exception as e:
        # Fallback: return original chunks
        return chunks


def score_chunks_for_faq(chunks: List[Dict]) -> List[Dict]:
    """
    Score each chunk for FAQ generation suitability
    """
    for chunk in chunks:
        content = chunk["content"]
        
        # Initialize scoring components
        scores = {
            "information_density": 0,
            "question_indicators": 0,
            "completeness": 0,
            "topic_clarity": 0,
            "definition_patterns": 0
        }
        
        # 1. Information Density Score
        # Chunks with good noun-to-word ratio often contain definable concepts
        words = content.split()
        if words:
            # Simple noun detection (words starting with capital letters, excluding first word)
            potential_nouns = sum(1 for w in words[1:] if w and w[0].isupper())
            scores["information_density"] = min(potential_nouns / len(words) * 10, 1.0)
        
        # 2. Question Indicators
        # Chunks that contain certain patterns often generate good FAQs
        question_patterns = [
            r'\b(what|why|how|when|where|who|which)\b',
            r'\b(is|are|can|should|must|will)\b',
            r'\b(definition|meaning|purpose|reason|cause)\b',
            r'\b(steps|process|procedure|method)\b',
            r'\b(example|instance|such as|like)\b',
            r'\b(important|significant|key|critical|essential)\b',
            r'\b(difference|comparison|versus|between)\b',
            r'\b(benefit|advantage|disadvantage|pros?|cons?)\b'
        ]
        
        pattern_matches = sum(
            1 for pattern in question_patterns
            if re.search(pattern, content.lower())
        )
        scores["question_indicators"] = min(pattern_matches / len(question_patterns), 1.0)
        
        # 3. Completeness Score
        # Chunks with complete sentences and proper structure
        sentences = re.split(r'[.!?]+', content)
        complete_sentences = sum(
            1 for s in sentences 
            if len(s.strip().split()) > 5  # At least 5 words
        )
        if sentences:
            scores["completeness"] = min(complete_sentences / len(sentences), 1.0)
        
        # 4. Topic Clarity
        # Chunks that focus on specific topics (not too broad)
        unique_words = set(word.lower() for word in words if len(word) > 3)
        if words:
            scores["topic_clarity"] = 1.0 - min(len(unique_words) / len(words), 1.0)
        
        # 5. Definition Patterns
        # Chunks containing definitions or explanations
        definition_patterns = [
            r'\b(is|are|refers to|defined as)\s+\w+',  # "X is a Y"
            r'\b(means|meaning|signifies)\b',         # "X means"
            r'\b(called|known as|termed)\b',          # "called/known as"
            r'\b(concept|idea|principle|theory)\b',    # Concept words
            r'\b(for example|e\.g\.|i\.e\.)\b',      # Examples
            r':\s*\n'                               # Colon followed by explanation
        ]
        
        pattern_matches = sum(
            1 for pattern in definition_patterns
            if re.search(pattern, content.lower())
        )
        scores["definition_patterns"] = min(pattern_matches / len(definition_patterns), 1.0)
        
        # Calculate final score (weighted average)
        weights = {
            "information_density": 0.25,
            "question_indicators": 0.25,
            "completeness": 0.2,
            "topic_clarity": 0.15,
            "definition_patterns": 0.15
        }
        
        faq_score = sum(score * weights[key] for key, score in scores.items())
        
        # Add scores to chunk
        chunk["faq_score"] = faq_score
        chunk["characteristics"] = scores
    
    return chunks


def select_diverse_chunks(
    chunks: List[Dict],
    max_chunks: int,
    diversity_weight: float = 0.7
) -> List[Dict]:
    """
    Select chunks that are both high-quality and diverse
    Uses Maximum Marginal Relevance (MMR) algorithm
    """
    if len(chunks) <= max_chunks:
        return sorted(chunks, key=lambda x: x["faq_score"], reverse=True)
    
    # Sort by FAQ score
    sorted_chunks = sorted(chunks, key=lambda x: x["faq_score"], reverse=True)
    
    # If diversity weight is 0, just return top chunks
    if diversity_weight == 0:
        return sorted_chunks[:max_chunks]
    
    # Extract text for similarity calculation
    texts = [chunk["content"] for chunk in sorted_chunks]
    
    # Calculate TF-IDF vectors
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except:
        # Fallback if TF-IDF fails
        return sorted_chunks[:max_chunks]
    
    # MMR selection
    selected_indices = []
    selected_chunks = []
    
    # Start with the highest scoring chunk
    selected_indices.append(0)
    selected_chunks.append(sorted_chunks[0])
    
    # Iteratively select diverse chunks
    while len(selected_chunks) < max_chunks and len(selected_indices) < len(sorted_chunks):
        candidate_indices = [
            i for i in range(len(sorted_chunks)) 
            if i not in selected_indices
        ]
        
        if not candidate_indices:
            break
        
        # Calculate MMR scores
        mmr_scores = []
        
        for idx in candidate_indices:
            # Relevance score (FAQ score)
            relevance = sorted_chunks[idx]["faq_score"]
            
            # Maximum similarity to already selected chunks
            similarities = []
            for selected_idx in selected_indices:
                sim = cosine_similarity(
                    tfidf_matrix[idx:idx+1],
                    tfidf_matrix[selected_idx:selected_idx+1]
                )[0][0]
                similarities.append(sim)
            
            max_similarity = max(similarities) if similarities else 0
            
            # MMR score
            mmr = (1 - diversity_weight) * relevance - diversity_weight * max_similarity
            mmr_scores.append((idx, mmr))
        
        # Select chunk with highest MMR
        best_idx = max(mmr_scores, key=lambda x: x[1])[0]
        selected_indices.append(best_idx)
        selected_chunks.append(sorted_chunks[best_idx])
    
    return selected_chunks


@router.get("/project-chat-query", response_model=List[str])
async def project_chat_query(
    request: Request,
    documentIds: List[str] = Query(..., description="List of document IDs for the project"),
    query: str = Query(..., description="User query for chat"),
):
    """
    Query document chunks for chat responses within a specific project.
    Optimized for conversational AI - returns minimal data for LLM processing.
    
    Fixed parameters:
    - maxChunks: 3
    - minChunkLength: 50
    - maxChunkLength: 1500
    
    Enhanced with relevance filtering to return empty results when no relevant content found.
    """
    
    # Fixed parameters
    MAX_CHUNKS = 3
    MIN_CHUNK_LENGTH = 50
    MAX_CHUNK_LENGTH = 1500
    
    # Enhanced relevance thresholds
    SIMILARITY_THRESHOLD = 0.7  # Vector similarity threshold (lower = more similar for cosine distance)
    RELEVANCE_THRESHOLD = 0.25  # Minimum relevance score (0.0-1.0)
    KEYWORD_OVERLAP_THRESHOLD = 0.1  # Minimum keyword overlap ratio
    
    logger.info(f"=== PROJECT CHAT QUERY START ===")
    logger.info(f"Query: '{query}'")
    logger.info(f"Document IDs: {documentIds}")
    logger.info(f"Thresholds - Similarity: {SIMILARITY_THRESHOLD}, Relevance: {RELEVANCE_THRESHOLD}, Keyword: {KEYWORD_OVERLAP_THRESHOLD}")
    
    try:
        # 1. Handle empty query
        if not query or query.strip() == "":
            logger.info("Empty query detected - returning empty results")
            return []
        
        search_query = query.strip()
        
        # 2. Get query embedding (fresh embedding every time)
        logger.debug("Getting fresh query embedding...")
        
        query_embedding = None
        
        # Method 1: Try vector store's embedding service
        if hasattr(vector_store, 'embeddings') and vector_store.embeddings:
            try:
                query_embedding = vector_store.embeddings.embed_query(search_query)
                logger.info("Successfully got embedding from vector store")
            except Exception as e:
                logger.error(f"Vector store embedding failed: {e}")
        
        # Method 2: Try from config
        if not query_embedding:
            try:
                from app.config import embeddings
                query_embedding = embeddings.embed_query(search_query)
                logger.info("Successfully got embedding from config")
            except Exception as e:
                logger.error(f"Config embedding failed: {e}")
        
        # Method 3: Direct OpenAI API call (fallback)
        if not query_embedding:
            try:
                import openai
                import os
                
                client = openai.AzureOpenAI(
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    api_version="2023-05-15",
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
                )
                
                response = client.embeddings.create(
                    input=search_query,
                    model="text-embedding-3-small"
                )
                
                query_embedding = response.data[0].embedding
                logger.info("Successfully got embedding from direct OpenAI API")
            except Exception as e:
                logger.error(f"Direct OpenAI embedding failed: {e}")
        
        # Final fallback
        if not query_embedding:
            logger.warning("All embedding methods failed, falling back to cached function")
            from app.routes.document_routes import get_cached_query_embedding
            query_embedding = get_cached_query_embedding(search_query)
        
        if not query_embedding:
            logger.error("Failed to get query embedding")
            return []
        
        logger.info(f"Query embedding obtained, length: {len(query_embedding)}")
        
        # 3. Perform similarity search
        logger.debug("Performing similarity search...")
        
        if isinstance(vector_store, AsyncPgVector):
            filter_dict = {"file_id": {"$in": documentIds}}
            
            documents_with_scores = await run_in_executor(
                None,
                vector_store.similarity_search_with_score_by_vector,
                query_embedding,
                k=MAX_CHUNKS * 5,  # Get more candidates for filtering
                filter=filter_dict
            )
        else:
            filter_dict = {"file_id": {"$in": documentIds}}
            
            documents_with_scores = vector_store.similarity_search_with_score_by_vector(
                query_embedding,
                k=MAX_CHUNKS * 5,
                filter=filter_dict
            )
        
        logger.info(f"Initial similarity search returned {len(documents_with_scores)} documents")
        
        # 4. Enhanced validation with similarity filtering
        validated_documents = []
        similarity_filtered_count = 0
        
        for idx, (doc, similarity_score) in enumerate(documents_with_scores):
            file_id = doc.metadata.get('file_id')
            
            # Check if document belongs to project
            if file_id not in documentIds:
                continue
                
            # Apply similarity threshold
            if similarity_score > SIMILARITY_THRESHOLD:
                similarity_filtered_count += 1
                logger.debug(f"Document {idx} filtered by similarity: score={similarity_score} > threshold={SIMILARITY_THRESHOLD}")
                continue
                
            validated_documents.append((doc, similarity_score))
            logger.debug(f"✅ Document {idx} passed similarity filter: file_id='{file_id}', score={similarity_score}")
        
        logger.info(f"After similarity filtering: {len(validated_documents)} documents (filtered: {similarity_filtered_count})")
        
        if not validated_documents:
            logger.info("❌ NO DOCUMENTS PASSED SIMILARITY FILTER - returning empty results")
            return []
        
        # 5. Process and score chunks with enhanced relevance filtering
        logger.debug("Processing and scoring chunks...")
        
        chat_chunks = []
        seen_content_hashes = set()
        length_filtered_count = 0
        duplicate_filtered_count = 0
        relevance_filtered_count = 0
        keyword_filtered_count = 0
        
        for idx, (document, similarity_score) in enumerate(validated_documents):
            content = document.page_content
            metadata = document.metadata or {}
            file_id = metadata.get('file_id', 'unknown')
            
            # Filter by length
            if not (MIN_CHUNK_LENGTH <= len(content) <= MAX_CHUNK_LENGTH):
                length_filtered_count += 1
                continue
            
            # Deduplicate based on content
            content_hash = get_content_hash(content)
            if content_hash in seen_content_hashes:
                duplicate_filtered_count += 1
                continue
            seen_content_hashes.add(content_hash)
            
            # Calculate enhanced relevance score
            relevance_metrics = calculate_enhanced_chat_relevance_score(
                content=content,
                query=query,
                similarity_score=similarity_score
            )
            
            # Apply relevance threshold
            if relevance_metrics["final_score"] < RELEVANCE_THRESHOLD:
                relevance_filtered_count += 1
                logger.debug(f"Chunk {idx} filtered by relevance: score={relevance_metrics['final_score']} < threshold={RELEVANCE_THRESHOLD}")
                continue
            
            # Apply keyword overlap threshold
            if relevance_metrics["keyword_overlap"] < KEYWORD_OVERLAP_THRESHOLD:
                keyword_filtered_count += 1
                logger.debug(f"Chunk {idx} filtered by keyword overlap: {relevance_metrics['keyword_overlap']} < threshold={KEYWORD_OVERLAP_THRESHOLD}")
                continue
            
            logger.debug(f"✅ Chunk {idx} passed all filters: relevance={relevance_metrics['final_score']}, keyword_overlap={relevance_metrics['keyword_overlap']}")
            
            chat_chunks.append({
                "content": content,
                "relevanceScore": relevance_metrics["final_score"],
                "similarity_score": similarity_score,
                "keyword_overlap": relevance_metrics["keyword_overlap"],
                "file_id": file_id,
                "metrics": relevance_metrics
            })
        
        logger.info(f"Chunk filtering summary:")
        logger.info(f"  Total retrieved: {len(validated_documents)}")
        logger.info(f"  Length filtered: {length_filtered_count}")
        logger.info(f"  Duplicate filtered: {duplicate_filtered_count}")
        logger.info(f"  Relevance filtered: {relevance_filtered_count}")
        logger.info(f"  Keyword filtered: {keyword_filtered_count}")
        logger.info(f"  Final chunks: {len(chat_chunks)}")
        
        if not chat_chunks:
            logger.info("❌ NO CHUNKS PASSED RELEVANCE FILTERING - returning empty results")
            return []
        
        # 6. Sort by relevance and select top chunks
        logger.debug("Sorting and selecting top chunks...")
        chat_chunks.sort(key=lambda x: x["relevanceScore"], reverse=True)
        selected_chunks = chat_chunks[:MAX_CHUNKS]
        
        logger.info(f"=== FINAL SELECTED CHUNKS FOR LLM ===")
        for idx, chunk in enumerate(selected_chunks):
            logger.info(f"Selected Chunk {idx}:")
            logger.info(f"  file_id: '{chunk['file_id']}'")
            logger.info(f"  relevance_score: {chunk['relevanceScore']}")
            logger.info(f"  similarity_score: {chunk['similarity_score']}")
            logger.info(f"  keyword_overlap: {chunk['keyword_overlap']}")
            logger.info(f"  content_length: {len(chunk['content'])}")
            logger.info(f"  content_preview: {chunk['content'][:200]}...")
        
        # 7. Format response (just content strings)
        result = [chunk["content"] for chunk in selected_chunks]
        
        logger.info(f"=== PROJECT CHAT QUERY SUCCESS: Returning {len(result)} relevant content strings ===")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Error in project chat query | Document IDs: %s | Query: %s | Error: %s | Traceback: %s",
            len(documentIds),
            query[:100] if query else "empty",
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=f"Project chat query failed: {str(e)}")


def calculate_enhanced_chat_relevance_score(content: str, query: str, similarity_score: float) -> Dict:
    """
    Enhanced relevance scoring with detailed metrics for better filtering.
    Returns a dictionary with individual scores and final combined score.
    """
    import re
    
    # Initialize metrics
    metrics = {
        "semantic_score": 0.0,
        "keyword_overlap": 0.0,
        "exact_phrase_match": 0.0,
        "qa_indicators": 0.0,
        "completeness": 0.0,
        "length_penalty": 0.0,
        "final_score": 0.0
    }
    
    query_lower = query.lower().strip()
    content_lower = content.lower()
    
    # 1. Semantic similarity score (from vector search)
    # Convert distance to similarity (lower distance = higher similarity)
    if similarity_score <= 1.0:
        metrics["semantic_score"] = max(0, 1.0 - similarity_score)
    else:
        metrics["semantic_score"] = 1.0 / (1.0 + similarity_score)
    
    # 2. Keyword overlap (most important for relevance)
    query_words = set(word.strip() for word in query_lower.split() if len(word.strip()) > 2)
    content_words = set(word.strip() for word in content_lower.split() if len(word.strip()) > 2)
    
    if query_words:
        overlap_count = len(query_words.intersection(content_words))
        metrics["keyword_overlap"] = overlap_count / len(query_words)
    else:
        metrics["keyword_overlap"] = 0.0
    
    # 3. Exact phrase matching (for multi-word queries)
    query_phrases = []
    if len(query.split()) > 1:
        # Extract 2-3 word phrases from query
        words = query_lower.split()
        for i in range(len(words) - 1):
            if i + 2 <= len(words):
                query_phrases.append(' '.join(words[i:i+2]))
            if i + 3 <= len(words):
                query_phrases.append(' '.join(words[i:i+3]))
    
    exact_matches = sum(1 for phrase in query_phrases if phrase in content_lower)
    metrics["exact_phrase_match"] = min(exact_matches / max(len(query_phrases), 1), 1.0) if query_phrases else 0.0
    
    # 4. Question-answering indicators
    qa_patterns = [
        r'\b(what|why|how|when|where|who|which)\b',
        r'\b(is|are|can|should|must|will|does|do)\b',
        r'\b(definition|meaning|purpose|explanation)\b',
        r'\b(example|for instance|such as)\b',
        r'\b(because|therefore|thus|hence|so)\b',
        r'\b(first|second|third|finally|then|next)\b',
        r'\b(important|key|main|primary|essential)\b'
    ]
    
    qa_score = 0
    for pattern in qa_patterns:
        if re.search(pattern, content_lower):
            qa_score += 1
    metrics["qa_indicators"] = min(qa_score / len(qa_patterns), 1.0)
    
    # 5. Content completeness (for standalone answers)
    completeness_indicators = [
        r'[.!?]\s+[A-Z]',  # Multiple sentences
        r'\b(however|but|although|while)\b',  # Contrasting information
        r'\b(additionally|furthermore|moreover|also)\b',  # Additional information
        r':\s*\n',  # Definitions or lists
        r'\b\d+[.)]\s',  # Numbered lists
        r'[•\-*]\s'  # Bullet points
    ]
    
    completeness_score = 0
    for pattern in completeness_indicators:
        if re.search(pattern, content):
            completeness_score += 1
    metrics["completeness"] = min(completeness_score / len(completeness_indicators), 1.0)
    
    # 6. Length penalty (optimal length for chat responses)
    optimal_length = 400
    length_diff = abs(len(content) - optimal_length)
    metrics["length_penalty"] = max(0.3, 1.0 - (length_diff / optimal_length))
    
    # 7. Calculate final score with emphasis on keyword relevance
    weights = {
        "semantic_score": 0.25,      # Vector similarity
        "keyword_overlap": 0.35,     # Most important - direct relevance
        "exact_phrase_match": 0.20,  # Phrase matching
        "qa_indicators": 0.10,       # QA suitability
        "completeness": 0.05,        # Content quality
        "length_penalty": 0.05       # Length optimization
    }
    
    metrics["final_score"] = sum(
        metrics[key] * weights[key] 
        for key in weights.keys()
    )
    
    # Round scores for readability
    for key in metrics:
        metrics[key] = round(metrics[key], 4)
    
    return metrics


# Debug endpoint to understand vector store structure
@router.get("/debug/vector-store-info")
async def get_vector_store_info(request: Request):
    """
    Debug endpoint to inspect vector store configuration
    """
    info = {
        "vector_store_type": str(type(vector_store)),
        "attributes": [attr for attr in dir(vector_store) if not attr.startswith('_')],
        "has_async_methods": hasattr(vector_store, 'asimilarity_search')
    }
    
    # Check for specific attributes
    if hasattr(vector_store, 'collection_name'):
        info['collection_name'] = vector_store.collection_name
    if hasattr(vector_store, 'table_name'):
        info['table_name'] = vector_store.table_name
    if hasattr(vector_store, 'embedding_function'):
        info['has_embedding_function'] = True
        
    return info


@router.get("/debug/test-relevance")
async def test_relevance_scoring(
    request: Request,
    query: str = Query(..., description="Test query"),
    content: str = Query(..., description="Test content"),
    similarity_score: float = Query(0.5, description="Test similarity score")
):
    """
    Debug endpoint to test relevance scoring algorithm
    """
    metrics = calculate_enhanced_chat_relevance_score(
        content=content,
        query=query,
        similarity_score=similarity_score
    )
    
    return {
        "query": query,
        "content_preview": content[:200] + "..." if len(content) > 200 else content,
        "content_length": len(content),
        "similarity_score": similarity_score,
        "relevance_metrics": metrics,
        "thresholds": {
            "similarity_threshold": 0.7,
            "relevance_threshold": 0.25,
            "keyword_overlap_threshold": 0.1
        },
        "passes_filters": {
            "similarity": similarity_score <= 0.7,
            "relevance": metrics["final_score"] >= 0.25,
            "keyword_overlap": metrics["keyword_overlap"] >= 0.1
        }
    }