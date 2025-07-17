import traceback
from typing import List, Dict, Optional
from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import hashlib
from langchain_core.runnables import run_in_executor

from app.config import logger, vector_store
from app.services.vector_store.async_pg_vector import AsyncPgVector

# Global configuration parameters
# FAQ generation parameters
FAQ_MAX_CHUNKS = 50
FAQ_MIN_CHUNK_LENGTH = 100
FAQ_MAX_CHUNK_LENGTH = 1000
FAQ_DIVERSITY_WEIGHT = 0.7
FAQ_SIMILARITY_THRESHOLD = 2.0  # More lenient for FAQ generation
FAQ_RELEVANCE_THRESHOLD = 0.1   # Lower threshold for FAQ
FAQ_KEYWORD_OVERLAP_THRESHOLD = 0.02  # Very low for FAQ

# Chat query parameters
CHAT_MAX_CHUNKS = 3
CHAT_MIN_CHUNK_LENGTH = 50
CHAT_MAX_CHUNK_LENGTH = 1500
CHAT_DIVERSITY_WEIGHT = 0.7  # Not used in chat mode but kept for consistency
CHAT_SIMILARITY_THRESHOLD = 1.2
CHAT_RELEVANCE_THRESHOLD = 0.20
CHAT_KEYWORD_OVERLAP_THRESHOLD = 0.05

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

async def process_document_chunks(
    document_ids: List[str],
    query: Optional[str] = None,
    max_chunks: int = 50,
    min_chunk_length: int = 100,
    max_chunk_length: int = 1000,
    diversity_weight: float = 0.7,
    similarity_threshold: float = 1.2,
    relevance_threshold: float = 0.20,
    keyword_overlap_threshold: float = 0.05,
    use_faq_scoring: bool = True
) -> List[ChunkResponse]:
    """
    Common function to process document chunks for both FAQ generation and chat queries.
    Always returns List[ChunkResponse] objects.
    
    Args:
        document_ids: List of document IDs to search in
        query: Search query (if None or empty, uses "document content" as fallback)
        max_chunks: Maximum number of chunks to return
        min_chunk_length: Minimum chunk length
        max_chunk_length: Maximum chunk length
        diversity_weight: Weight for diversity vs relevance (0-1)
        similarity_threshold: Vector similarity threshold
        relevance_threshold: Minimum relevance score
        keyword_overlap_threshold: Minimum keyword overlap ratio
        use_faq_scoring: Whether to use FAQ scoring or chat relevance scoring
    
    Returns:
        List of ChunkResponse objects
    """
    
    # Handle empty query
    if not query or query.strip() == "":
        query = "document content"
        logger.info(f"Empty query detected - using fallback: '{query}'")
    
    search_query = query.strip()
    
    logger.info(f"Processing document chunks:")
    logger.info(f"  Query: '{search_query}'")
    logger.info(f"  Document IDs: {document_ids}")
    logger.info(f"  Max chunks: {max_chunks}")
    logger.info(f"  Use FAQ scoring: {use_faq_scoring}")
    
    try:
        # 1. Get query embedding
        query_embedding = await get_query_embedding(search_query)
        if not query_embedding:
            logger.error("Failed to get query embedding")
            return []
        
        # 2. Perform similarity search
        documents_with_scores = await perform_similarity_search(
            query_embedding=query_embedding,
            document_ids=document_ids,
            max_candidates=max_chunks * 5
        )
        
        logger.info(f"Initial similarity search returned {len(documents_with_scores)} documents")
        
        # 3. Filter and validate documents
        validated_documents = filter_by_similarity_threshold(
            documents_with_scores=documents_with_scores,
            document_ids=document_ids,
            similarity_threshold=similarity_threshold
        )
        
        logger.info(f"After similarity filtering: {len(validated_documents)} documents")
        
        if not validated_documents:
            logger.info("❌ NO DOCUMENTS PASSED SIMILARITY FILTER - returning empty results")
            return []
        
        # 4. Process chunks based on scoring method
        if use_faq_scoring:
            processed_chunks = await process_chunks_for_faq(
                validated_documents=validated_documents,
                min_chunk_length=min_chunk_length,
                max_chunk_length=max_chunk_length,
                diversity_weight=diversity_weight,
                max_chunks=max_chunks
            )
        else:
            processed_chunks = await process_chunks_for_chat(
                validated_documents=validated_documents,
                query=search_query,
                min_chunk_length=min_chunk_length,
                max_chunk_length=max_chunk_length,
                relevance_threshold=relevance_threshold,
                keyword_overlap_threshold=keyword_overlap_threshold,
                max_chunks=max_chunks
            )
        
        logger.info(f"Final processed chunks: {len(processed_chunks)}")
        
        # 5. Convert to ChunkResponse objects
        chunk_responses = []
        for chunk in processed_chunks:
            chunk_response = ChunkResponse(
                chunkId=chunk["id"],
                content=chunk["content"],
                metadata=chunk["metadata"],
                score=chunk.get("faq_score", chunk.get("relevanceScore", 0)),
                characteristics=chunk.get("characteristics", chunk.get("metrics", {}))
            )
            chunk_responses.append(chunk_response)
        
        return chunk_responses
        
    except Exception as e:
        logger.error(f"Error processing document chunks: {str(e)}")
        raise


async def get_query_embedding(query: str):
    """Get embedding for the query using available embedding services."""
    query_embedding = None
    
    # Method 1: Try vector store's embedding service
    if hasattr(vector_store, 'embeddings') and vector_store.embeddings:
        try:
            query_embedding = vector_store.embeddings.embed_query(query)
            logger.info("Successfully got embedding from vector store")
        except Exception as e:
            logger.error(f"Vector store embedding failed: {e}")
    
    # Method 2: Try from config
    if not query_embedding:
        try:
            from app.config import embeddings
            query_embedding = embeddings.embed_query(query)
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
                input=query,
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
        query_embedding = get_cached_query_embedding(query)
    
    return query_embedding


async def perform_similarity_search(query_embedding, document_ids: List[str], max_candidates: int):
    """Perform similarity search using the vector store."""
    filter_dict = {"file_id": {"$in": document_ids}}
    
    if isinstance(vector_store, AsyncPgVector):
        documents_with_scores = await run_in_executor(
            None,
            vector_store.similarity_search_with_score_by_vector,
            query_embedding,
            max_candidates,
            filter_dict
        )
    else:
        documents_with_scores = vector_store.similarity_search_with_score_by_vector(
            query_embedding,
            k=max_candidates,
            filter=filter_dict
        )
    
    return documents_with_scores


def filter_by_similarity_threshold(documents_with_scores, document_ids: List[str], similarity_threshold: float):
    """Filter documents by similarity threshold."""
    validated_documents = []
    similarity_filtered_count = 0
    
    for idx, (doc, similarity_score) in enumerate(documents_with_scores):
        file_id = doc.metadata.get('file_id')
        
        # Check if document belongs to project
        if file_id not in document_ids:
            continue
            
        # Apply similarity threshold
        if similarity_score > similarity_threshold:
            similarity_filtered_count += 1
            logger.debug(f"Document {idx} filtered by similarity: score={similarity_score} > threshold={similarity_threshold}")
            continue
            
        validated_documents.append((doc, similarity_score))
        logger.debug(f"✅ Document {idx} passed similarity filter: file_id='{file_id}', score={similarity_score}")
    
    logger.info(f"Similarity filtering: {len(validated_documents)} documents passed (filtered: {similarity_filtered_count})")
    return validated_documents


async def process_chunks_for_faq(validated_documents, min_chunk_length: int, max_chunk_length: int, 
                                diversity_weight: float, max_chunks: int):
    """Process chunks for FAQ generation with diversity scoring."""
    # 1. Get all chunks
    all_chunks = []
    for idx, (document, similarity_score) in enumerate(validated_documents):
        file_id = document.metadata.get('file_id', 'unknown')
        all_chunks.append({
            "id": f"{file_id}_{idx}",
            "content": document.page_content,
            "metadata": document.metadata,
            "similarity_score": similarity_score
        })
    
    # 2. Deduplicate chunks
    unique_chunks = deduplicate_chunks(all_chunks)
    
    # 3. Filter chunks by length
    filtered_chunks = [
        chunk for chunk in unique_chunks
        if min_chunk_length <= len(chunk["content"]) <= max_chunk_length
    ]
    
    if not filtered_chunks:
        filtered_chunks = unique_chunks
    
    # 4. Score chunks for FAQ suitability
    scored_chunks = score_chunks_for_faq(filtered_chunks)
    
    # 5. Select diverse, high-quality chunks
    selected_chunks = select_diverse_chunks(
        scored_chunks,
        max_chunks=max_chunks,
        diversity_weight=diversity_weight
    )
    
    return selected_chunks


async def process_chunks_for_chat(validated_documents, query: str, min_chunk_length: int, 
                                 max_chunk_length: int, relevance_threshold: float,
                                 keyword_overlap_threshold: float, max_chunks: int):
    """Process chunks for chat queries with relevance scoring."""
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
        if not (min_chunk_length <= len(content) <= max_chunk_length):
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
        if relevance_metrics["final_score"] < relevance_threshold:
            relevance_filtered_count += 1
            logger.debug(f"Chunk {idx} filtered by relevance: score={relevance_metrics['final_score']} < threshold={relevance_threshold}")
            continue
        
        # Apply keyword overlap threshold
        if relevance_metrics["keyword_overlap"] < keyword_overlap_threshold:
            keyword_filtered_count += 1
            logger.debug(f"Chunk {idx} filtered by keyword overlap: {relevance_metrics['keyword_overlap']} < threshold={keyword_overlap_threshold}")
            continue
        
        logger.debug(f"✅ Chunk {idx} passed all filters: relevance={relevance_metrics['final_score']}, keyword_overlap={relevance_metrics['keyword_overlap']}")
        
        chat_chunks.append({
            "id": f"{file_id}_{idx}",
            "content": content,
            "metadata": metadata,
            "relevanceScore": relevance_metrics["final_score"],
            "similarity_score": similarity_score,
            "keyword_overlap": relevance_metrics["keyword_overlap"],
            "file_id": file_id,
            "metrics": relevance_metrics
        })
    
    logger.info(f"Chat chunk filtering summary:")
    logger.info(f"  Total retrieved: {len(validated_documents)}")
    logger.info(f"  Length filtered: {length_filtered_count}")
    logger.info(f"  Duplicate filtered: {duplicate_filtered_count}")
    logger.info(f"  Relevance filtered: {relevance_filtered_count}")
    logger.info(f"  Keyword filtered: {keyword_filtered_count}")
    logger.info(f"  Final chunks: {len(chat_chunks)}")
    
    # Sort by relevance and select top chunks
    chat_chunks.sort(key=lambda x: x["relevanceScore"], reverse=True)
    selected_chunks = chat_chunks[:max_chunks]
    
    return selected_chunks


@router.get("/document-chunks", response_model=List[str])
async def get_document_chunks(
    request: Request,
    documentIds: List[str] = Query(..., description="List of document IDs"),
    maxChunks: int = Query(FAQ_MAX_CHUNKS, description="Maximum number of chunks to return"),
    minChunkLength: int = Query(FAQ_MIN_CHUNK_LENGTH, description="Minimum chunk length"),
    maxChunkLength: int = Query(FAQ_MAX_CHUNK_LENGTH, description="Maximum chunk length"),
    diversityWeight: float = Query(FAQ_DIVERSITY_WEIGHT, description="Balance between relevance and diversity"),
    query: Optional[str] = Query(None, description="Optional query to filter chunks")
):
    """
    Extract the best document chunks using a hybrid strategy optimized for FAQ generation.
    Returns List[str] of content strings.
    Accepts empty query and uses "document content" as fallback.
    """
    
    try:
        chunk_responses = await process_document_chunks(
            document_ids=documentIds,
            query=query,
            max_chunks=maxChunks,
            min_chunk_length=minChunkLength,
            max_chunk_length=maxChunkLength,
            diversity_weight=diversityWeight,
            similarity_threshold=FAQ_SIMILARITY_THRESHOLD,
            relevance_threshold=FAQ_RELEVANCE_THRESHOLD,
            keyword_overlap_threshold=FAQ_KEYWORD_OVERLAP_THRESHOLD,
            use_faq_scoring=True
        )
        
        if not chunk_responses:
            raise HTTPException(status_code=404, detail="No relevant chunks found for the specified documents")
        
        # Extract content strings from ChunkResponse objects
        content_strings = [chunk.content for chunk in chunk_responses]
        return content_strings
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_document_chunks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/project-chat-query", response_model=List[str])
async def project_chat_query(
    request: Request,
    documentIds: List[str] = Query(..., description="List of document IDs for the project"),
    query: str = Query(..., description="User query for chat"),
):
    """
    Query document chunks for chat responses within a specific project.
    Optimized for conversational AI - returns minimal data for LLM processing.
    
    Uses global chat configuration parameters:
    - maxChunks: CHAT_MAX_CHUNKS (3)
    - minChunkLength: CHAT_MIN_CHUNK_LENGTH (50)
    - maxChunkLength: CHAT_MAX_CHUNK_LENGTH (1500)
    
    Enhanced with relevance filtering to return empty results when no relevant content found.
    """
    
    logger.info(f"=== PROJECT CHAT QUERY START ===")
    logger.info(f"Query: '{query}'")
    logger.info(f"Document IDs: {documentIds}")
    
    try:
        chunk_responses = await process_document_chunks(
            document_ids=documentIds,
            query=query,
            max_chunks=CHAT_MAX_CHUNKS,
            min_chunk_length=CHAT_MIN_CHUNK_LENGTH,
            max_chunk_length=CHAT_MAX_CHUNK_LENGTH,
            diversity_weight=CHAT_DIVERSITY_WEIGHT,
            similarity_threshold=CHAT_SIMILARITY_THRESHOLD,
            relevance_threshold=CHAT_RELEVANCE_THRESHOLD,
            keyword_overlap_threshold=CHAT_KEYWORD_OVERLAP_THRESHOLD,
            use_faq_scoring=False
        )
        
        # Extract content strings from ChunkResponse objects
        content_strings = [chunk.content for chunk in chunk_responses]
        
        logger.info(f"=== PROJECT CHAT QUERY SUCCESS: Returning {len(content_strings)} relevant content strings ===")
        return content_strings
        
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