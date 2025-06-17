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
    diversityWeight: float = Query(0.7, description="Balance between relevance and diversity")
):
    """
    Extract the best document chunks using a hybrid strategy:
    1. Information density (good for questions)
    2. Diversity (covers different topics)
    3. Completeness (self-contained chunks)
    4. Question indicators (chunks that naturally prompt questions)
    
    These chunks can be used by other services for FAQ generation.
    """
    
    logger.debug(f"Received GET request with params: documentIds={documentIds}, maxChunks={maxChunks}, minChunkLength={minChunkLength}, maxChunkLength={maxChunkLength}, diversityWeight={diversityWeight}")
    
    try:
        # 1. Get all chunks for the specified documents
        all_chunks = await get_all_chunks_for_documents(document_ids=documentIds)
        
        if not all_chunks:
            raise HTTPException(status_code=404, detail="No chunks found for the specified files")

        # 2. Deduplicate chunks FIRST
        unique_chunks = deduplicate_chunks(all_chunks)
        logger.info(f"Reduced {len(all_chunks)} chunks to {len(unique_chunks)} unique chunks")
        
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
        logger.error(
            "Error retrieving document chunks | Document IDs: %s | Error: %s | Traceback: %s",
            documentIds,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


# Optional: Add POST endpoint for backwards compatibility if needed
@router.post("/document-chunks", response_model=List[ChunkResponse])
async def post_document_chunks(
    request: Request,
    chunk_request: ChunkExtractionRequest
):
    """
    POST version of document chunks extraction for backwards compatibility.
    Delegates to the GET version with extracted parameters.
    """
    logger.debug(f"Received POST request with body: {chunk_request}")
    
    # Delegate to the GET endpoint logic
    return await get_document_chunks(
        request=request,
        documentIds=chunk_request.documentIds,
        maxChunks=chunk_request.maxChunks,
        minChunkLength=chunk_request.minChunkLength,
        maxChunkLength=chunk_request.maxChunkLength,
        diversityWeight=chunk_request.diversityWeight
    )


async def get_all_chunks_for_documents(document_ids: List[str]) -> List[Dict]:
    """
    Retrieve all chunks for given document_ids with metadata
    """
    chunks = []
    
    logger.debug(f"Starting chunk retrieval for document_ids: {document_ids}")
    
    try:
        # For AsyncPgVector, we should use its methods properly
        if isinstance(vector_store, AsyncPgVector):
            logger.info(f"Using AsyncPgVector query for document_ids: {document_ids}")
            
            # Query based only on file_id
            filter_dict = {
                "file_id": {"$in": document_ids}
            }
            
            logger.debug(f"Using filter: {filter_dict}")
            
            # Instead of empty string, use a common word to avoid empty embedding
            all_docs = await run_in_executor(
                None,
                lambda: vector_store.similarity_search_with_score(
                    query="document content text",  # More descriptive query
                    k=10000,      # High number to get all
                    filter=filter_dict
                )
            )
            
            logger.debug(f"Vector search returned {len(all_docs)} documents")
            
            # Process results
            seen_contents = set()  # Track unique content
            for idx, (doc, score) in enumerate(all_docs):
                file_id = doc.metadata.get('file_id')
                
                logger.debug(f"Processing document {idx}: file_id='{file_id}', score={score}")
                if idx == 0:  # Log first document metadata for debugging
                    logger.debug(f"Sample document metadata: {doc.metadata}")
                    logger.debug(f"Content preview: {doc.page_content[:100]}...")
                
                # Skip duplicates based on content
                content_hash = get_content_hash(doc.page_content)
                if content_hash in seen_contents:
                    logger.debug(f"Skipping duplicate content for file_id: {file_id}")
                    continue
                seen_contents.add(content_hash)
                
                if file_id in document_ids:
                    chunks.append({
                        "id": f"{file_id}_{idx}",
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "embedding": None
                    })
                    logger.info(f"Found matching chunk for file_id: {file_id}, content length: {len(doc.page_content)}")
                else:
                    logger.debug(f"file_id '{file_id}' not in target document_ids {document_ids}")
                    
        else:
            # For other vector stores
            logger.info(f"Using standard similarity search for document_ids: {document_ids}")
            filter_dict = {
                "file_id": {"$in": document_ids}
            }
            
            all_docs = vector_store.similarity_search_with_score(
                query="document content text",
                k=10000,
                filter=filter_dict
            )
            
            logger.debug(f"Standard search returned {len(all_docs)} documents")
            
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
                    logger.info(f"Found matching chunk for file_id: {file_id}")
                    
    except Exception as e:
        logger.error(f"Error in get_all_chunks_for_documents: {str(e)}\n{traceback.format_exc()}")
        raise
    
    if not chunks:
        logger.warning(f"No chunks found for document_ids {document_ids}")
        logger.warning("Check if the documents were properly embedded in the vector store")
    else:
        logger.info(f"Found {len(chunks)} unique chunks for document_ids {document_ids}")
    
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
        
        logger.info(f"Deduplication: {len(chunks)} -> {len(unique_chunks)} chunks")
        return unique_chunks
        
    except Exception as e:
        logger.error(f"Error in deduplication: {str(e)}")
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
    """
    
    # Fixed parameters
    MAX_CHUNKS = 3
    MIN_CHUNK_LENGTH = 50
    MAX_CHUNK_LENGTH = 1500
    
    logger.info(f"=== PROJECT CHAT QUERY START ===")
    logger.info(f"Query: '{query}'")
    logger.info(f"Document IDs: {documentIds}")
    logger.info(f"Document count: {len(documentIds)}")
    
    try:
        # 1. Get query embedding (reuse existing cached function)
        logger.debug("Step 1: Getting query embedding...")
        query_embedding = get_cached_query_embedding(query)
        logger.info(f"Query embedding obtained, length: {len(query_embedding) if query_embedding else 'None'}")
        
        # 2. Perform similarity search across project files
        logger.debug("Step 2: Performing similarity search...")
        logger.info(f"Vector store type: {type(vector_store)}")
        logger.info(f"Using filter: custom_id in {documentIds}")
        
        if isinstance(vector_store, AsyncPgVector):
            logger.debug("Using AsyncPgVector similarity search")
            # Use the indexed custom_id column for efficient filtering
            documents_with_scores = await run_in_executor(
                None,
                vector_store.similarity_search_with_score_by_vector,
                query_embedding,
                k=MAX_CHUNKS * 3,  # Get more initially for filtering
                filter={"custom_id": {"$in": documentIds}}
            )
        else:
            logger.debug("Using standard vector store similarity search")
            documents_with_scores = vector_store.similarity_search_with_score_by_vector(
                query_embedding,
                k=MAX_CHUNKS * 3,
                filter={"custom_id": {"$in": documentIds}}
            )
        
        logger.info(f"Similarity search returned {len(documents_with_scores)} documents")
        
        if not documents_with_scores:
            logger.warning(f"No chunks found for project documentIds: {documentIds}")
            logger.warning("This could mean:")
            logger.warning("1. Documents weren't embedded with these custom_ids")
            logger.warning("2. Vector store filter isn't working correctly")
            logger.warning("3. Documents exist but don't match the query")
            return []
        
        # Log first few results for debugging
        for i, (doc, score) in enumerate(documents_with_scores[:3]):
            logger.debug(f"Document {i}: custom_id='{doc.metadata.get('custom_id', 'NOT_SET')}', "
                        f"file_id='{doc.metadata.get('file_id', 'NOT_SET')}', "
                        f"score={score}, content_length={len(doc.page_content)}")
            logger.debug(f"Content preview: {doc.page_content[:100]}...")
        
        # 3. Process and score chunks for chat suitability
        logger.debug("Step 3: Processing and scoring chunks...")
        chat_chunks = []
        seen_content_hashes = set()
        length_filtered_count = 0
        duplicate_filtered_count = 0
        
        for idx, (document, similarity_score) in enumerate(documents_with_scores):
            content = document.page_content
            metadata = document.metadata or {}
            custom_id = metadata.get('custom_id', 'unknown')
            file_id = metadata.get('file_id', 'unknown')
            
            logger.debug(f"Processing chunk {idx}: custom_id='{custom_id}', file_id='{file_id}', "
                        f"content_length={len(content)}, similarity_score={similarity_score}")
            
            # Filter by length
            if not (MIN_CHUNK_LENGTH <= len(content) <= MAX_CHUNK_LENGTH):
                length_filtered_count += 1
                logger.debug(f"Chunk {idx} filtered by length: {len(content)} not in range [{MIN_CHUNK_LENGTH}, {MAX_CHUNK_LENGTH}]")
                continue
            
            # Deduplicate based on content
            content_hash = get_content_hash(content)
            if content_hash in seen_content_hashes:
                duplicate_filtered_count += 1
                logger.debug(f"Chunk {idx} filtered as duplicate content")
                continue
            seen_content_hashes.add(content_hash)
            
            # Calculate chat relevance score
            chat_score = calculate_chat_relevance_score(
                content=content,
                query=query,
                similarity_score=similarity_score
            )
            
            logger.debug(f"Chunk {idx} scored: relevance_score={chat_score}")
            
            chat_chunks.append({
                "content": content,
                "relevanceScore": chat_score,
                "similarity_score": similarity_score,  # Keep for sorting
                "custom_id": custom_id,
                "file_id": file_id
            })
        
        logger.info(f"Chunk processing summary:")
        logger.info(f"  Total retrieved: {len(documents_with_scores)}")
        logger.info(f"  Length filtered: {length_filtered_count}")
        logger.info(f"  Duplicate filtered: {duplicate_filtered_count}")
        logger.info(f"  Final chunks: {len(chat_chunks)}")
        
        if not chat_chunks:
            logger.warning("No chunks passed filtering! Check length and duplication filters.")
            return []
        
        # 4. Sort by relevance and select top chunks
        logger.debug("Step 4: Sorting and selecting top chunks...")
        chat_chunks.sort(key=lambda x: x["relevanceScore"], reverse=True)
        selected_chunks = chat_chunks[:MAX_CHUNKS]
        
        logger.info(f"Selected {len(selected_chunks)} top chunks:")
        for i, chunk in enumerate(selected_chunks):
            logger.info(f"  Chunk {i}: relevance_score={chunk['relevanceScore']}, "
                       f"custom_id='{chunk['custom_id']}', content_length={len(chunk['content'])}")
            logger.debug(f"  Content preview: {chunk['content'][:100]}...")
        
        # 5. Format response (just content strings)
        result = [chunk["content"] for chunk in selected_chunks]
        logger.info(f"=== PROJECT CHAT QUERY SUCCESS: Returning {len(result)} content strings ===")
        return result
        
    except HTTPException:
        logger.error("HTTPException in project_chat_query")
        raise
    except Exception as e:
        logger.error(
            "Error in project chat query | Document IDs: %s | Query: %s | Error: %s | Traceback: %s",
            len(documentIds),
            query[:100],
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=f"Project chat query failed: {str(e)}")


def calculate_chat_relevance_score(content: str, query: str, similarity_score: float) -> float:
    """
    Calculate relevance score specifically for chat responses.
    Combines semantic similarity with chat-specific factors.
    """
    
    # Base score from semantic similarity (0.0 to 1.0, higher is better)
    # Note: similarity_score from vector search is distance (lower is better)
    # Convert to similarity score (higher is better)
    semantic_score = max(0, 1.0 - similarity_score) if similarity_score <= 1.0 else 1.0 / (1.0 + similarity_score)
    
    # Chat-specific scoring factors
    query_lower = query.lower()
    content_lower = content.lower()
    
    # 1. Direct keyword overlap
    query_words = set(query_lower.split())
    content_words = set(content_lower.split())
    keyword_overlap = len(query_words.intersection(content_words)) / max(len(query_words), 1)
    
    # 2. Question answering indicators
    qa_indicators = [
        r'\b(what|why|how|when|where|who|which)\b',
        r'\b(is|are|can|should|must|will|does|do)\b',
        r'\b(definition|meaning|purpose|explanation)\b',
        r'\b(example|for instance|such as)\b',
        r'\b(because|therefore|thus|hence|so)\b',
        r'\b(first|second|third|finally|then|next)\b',  # Step indicators
        r'\b(important|key|main|primary|essential)\b'
    ]
    
    qa_score = 0
    for pattern in qa_indicators:
        if re.search(pattern, content_lower):
            qa_score += 1
    qa_score = min(qa_score / len(qa_indicators), 1.0)
    
    # 3. Completeness indicators (good for standalone answers)
    completeness_patterns = [
        r'[.!?]\s+[A-Z]',  # Multiple sentences
        r'\b(however|but|although|while)\b',  # Contrasting information
        r'\b(additionally|furthermore|moreover|also)\b',  # Additional information
        r':\s*\n',  # Definitions or lists
        r'\b\d+[.)]\s',  # Numbered lists
        r'[•\-*]\s'  # Bullet points
    ]
    
    completeness_score = 0
    for pattern in completeness_patterns:
        if re.search(pattern, content):
            completeness_score += 1
    completeness_score = min(completeness_score / len(completeness_patterns), 1.0)
    
    # 4. Length penalty for very short or very long chunks
    optimal_length = 400  # Target chunk length for chat (reduced from 800)
    length_penalty = 1.0 - abs(len(content) - optimal_length) / optimal_length
    length_penalty = max(0.5, length_penalty)  # Don't penalize too heavily
    
    # Combine scores with weights optimized for chat
    final_score = (
        semantic_score * 0.4 +           # Semantic similarity is most important
        keyword_overlap * 0.25 +         # Direct keyword match
        qa_score * 0.20 +               # QA indicators
        completeness_score * 0.10 +     # Completeness
        length_penalty * 0.05           # Length optimization
    )
    
    return round(final_score, 4)