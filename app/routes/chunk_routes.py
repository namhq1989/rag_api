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