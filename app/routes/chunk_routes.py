import traceback
from typing import List, Dict, Optional
from fastapi import APIRouter, HTTPException, Request, Body
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
    file_ids: List[str]
    max_chunks: int = 50
    min_chunk_length: int = 100
    max_chunk_length: int = 1000
    diversity_weight: float = 0.7  # Balance between relevance and diversity

class ChunkResponse(BaseModel):
    chunk_id: str
    content: str
    metadata: Dict
    score: float
    characteristics: Dict  # Why this chunk was selected

router = APIRouter()

@router.post("/extract-faq-chunks", response_model=List[ChunkResponse])
async def extract_faq_chunks(
    request: Request,
    body: ChunkExtractionRequest = Body(...)
):
    """
    Extract the best chunks for FAQ generation using a hybrid strategy:
    1. Information density (good for questions)
    2. Diversity (covers different topics)
    3. Completeness (self-contained chunks)
    4. Question indicators (chunks that naturally prompt questions)
    """
    
    user_id = request.state.user.get("id", "public") if hasattr(request.state, "user") else "public"
    
    try:
        # 1. Get all chunks for the specified documents
        all_chunks = await get_all_chunks_for_files(
            user_id=user_id,
            file_ids=body.file_ids
        )
        
        if not all_chunks:
            raise HTTPException(status_code=404, detail="No chunks found for the specified files")

        # 2. Deduplicate chunks FIRST
        unique_chunks = deduplicate_chunks(all_chunks)
        logger.info(f"Reduced {len(all_chunks)} chunks to {len(unique_chunks)} unique chunks")
        
        # 3. Filter chunks by length
        filtered_chunks = [
            chunk for chunk in unique_chunks
            if body.min_chunk_length <= len(chunk["content"]) <= body.max_chunk_length
        ]
        
        if not filtered_chunks:
            filtered_chunks = unique_chunks
        
        # 4. Score chunks for FAQ suitability
        scored_chunks = score_chunks_for_faq(filtered_chunks)
        
        # 5. Select diverse, high-quality chunks
        selected_chunks = select_diverse_chunks(
            scored_chunks,
            max_chunks=body.max_chunks,
            diversity_weight=body.diversity_weight
        )
        
        # 6. Format response
        return [
            ChunkResponse(
                chunk_id=chunk["id"],
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
            "Error extracting FAQ chunks | File IDs: %s | Error: %s | Traceback: %s",
            body.file_ids,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


async def get_all_chunks_for_files(
    user_id: str,
    file_ids: List[str]
) -> List[Dict]:
    """
    Retrieve all chunks for given file_ids with metadata
    """
    chunks = []
    
    try:
        # For AsyncPgVector, we should use its methods properly
        if isinstance(vector_store, AsyncPgVector):
            logger.info(f"Using AsyncPgVector query for file_ids: {file_ids}")
            
            # AsyncPgVector has a method to get all documents
            # We can use similarity_search but with a high k value
            filter_dict = {
                "file_id": {"$in": file_ids},
                "user_id": user_id
            }
            
            # Instead of empty string, use a common word to avoid empty embedding
            # This is a workaround but more efficient than direct SQL
            all_docs = await run_in_executor(
                None,
                lambda: vector_store.similarity_search_with_score(
                    query="the",  # Common word instead of empty string
                    k=10000,      # High number to get all
                    filter=filter_dict
                )
            )
            
            # Process results
            seen_contents = set()  # Track unique content
            for idx, (doc, score) in enumerate(all_docs):
                doc_file_id = doc.metadata.get('file_id')
                
                # Skip duplicates based on content
                content_hash = get_content_hash(doc.page_content)
                if content_hash in seen_contents:
                    continue
                seen_contents.add(content_hash)
                
                if doc_file_id in file_ids:
                    chunks.append({
                        "id": f"{doc_file_id}_{idx}",
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "embedding": None
                    })
                    
        else:
            # For other vector stores
            logger.info(f"Using standard similarity search for file_ids: {file_ids}")
            filter_dict = {
                "file_id": {"$in": file_ids},
                "user_id": user_id
            }
            
            all_docs = vector_store.similarity_search_with_score(
                query="the",  # Common word
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
                
                doc_file_id = doc.metadata.get('file_id')
                if doc_file_id in file_ids:
                    chunks.append({
                        "id": f"{doc_file_id}_{idx}",
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "embedding": None
                    })
                    
    except Exception as e:
        logger.error(f"Error in get_all_chunks_for_files: {str(e)}\n{traceback.format_exc()}")
        raise
    
    if not chunks:
        logger.warning(f"No chunks found for file_ids {file_ids} and user_id {user_id}")
    else:
        logger.info(f"Found {len(chunks)} unique chunks for file_ids {file_ids}")
    
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


@router.post("/analyze-chunks")
async def analyze_chunks_for_faq(
    request: Request,
    body: ChunkExtractionRequest = Body(...)
):
    """
    Analyze chunks and provide statistics about FAQ suitability
    """
    user_id = request.state.user.get("id", "public") if hasattr(request.state, "user") else "public"
    
    try:
        # Get all chunks
        all_chunks = await get_all_chunks_for_files(
            user_id=user_id,
            file_ids=body.file_ids
        )
        
        if not all_chunks:
            return {
                "total_chunks": 0,
                "scored_chunks": 0,
                "score_distribution": {},
                "characteristics_analysis": {},
                "recommended_chunk_count": 0,
                "message": "No chunks found for the specified files"
            }
        
        # Deduplicate first
        unique_chunks = deduplicate_chunks(all_chunks)
        
        # Score chunks
        scored_chunks = score_chunks_for_faq(unique_chunks)
        
        # Calculate statistics
        scores = [chunk["faq_score"] for chunk in scored_chunks]
        characteristics_avg = {}
        
        if scored_chunks:
            for key in scored_chunks[0]["characteristics"]:
                values = [chunk["characteristics"][key] for chunk in scored_chunks]
                characteristics_avg[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values))
                }
        
        return {
            "total_chunks": len(all_chunks),
            "unique_chunks": len(unique_chunks),
            "scored_chunks": len(scored_chunks),
            "score_distribution": {
                "mean": float(np.mean(scores)) if scores else 0,
                "std": float(np.std(scores)) if scores else 0,
                "min": float(np.min(scores)) if scores else 0,
                "max": float(np.max(scores)) if scores else 0,
                "percentiles": {
                    "25": float(np.percentile(scores, 25)) if scores else 0,
                    "50": float(np.percentile(scores, 50)) if scores else 0,
                    "75": float(np.percentile(scores, 75)) if scores else 0,
                    "90": float(np.percentile(scores, 90)) if scores else 0
                }
            },
            "characteristics_analysis": characteristics_avg,
            "recommended_chunk_count": min(
                max(10, int(len(scored_chunks) * 0.2)),  # 20% of chunks
                50  # Cap at 50
            ),
            "duplicate_ratio": 1 - (len(unique_chunks) / len(all_chunks)) if all_chunks else 0
        }
        
    except Exception as e:
        logger.error(
            "Error analyzing chunks | File IDs: %s | Error: %s | Traceback: %s",
            body.file_ids,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


# Debug endpoint to understand vector store structure
@router.get("/debug/vector-store-info")
async def get_vector_store_info(request: Request):
    """
    Debug endpoint to inspect vector store configuration
    """
    user_id = request.state.user.get("id", "public") if hasattr(request.state, "user") else "public"
    
    info = {
        "vector_store_type": str(type(vector_store)),
        "attributes": [attr for attr in dir(vector_store) if not attr.startswith('_')],
        "has_async_methods": hasattr(vector_store, 'asimilarity_search'),
        "user_id": user_id
    }
    
    # Check for specific attributes
    if hasattr(vector_store, 'collection_name'):
        info['collection_name'] = vector_store.collection_name
    if hasattr(vector_store, 'table_name'):
        info['table_name'] = vector_store.table_name
    if hasattr(vector_store, 'embedding_function'):
        info['has_embedding_function'] = True
        
    return info