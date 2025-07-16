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
    
    # logger.debug(f"Received GET request with params: documentIds={documentIds}, maxChunks={maxChunks}, minChunkLength={minChunkLength}, maxChunkLength={maxChunkLength}, diversityWeight={diversityWeight}")
    
    try:
        # 1. Get all chunks for the specified documents
        all_chunks = await get_all_chunks_for_documents(document_ids=documentIds, query=query)
        
        if not all_chunks:
            raise HTTPException(status_code=404, detail="No chunks found for the specified files")

        # 2. Deduplicate chunks FIRST
        unique_chunks = deduplicate_chunks(all_chunks)
        # logger.info(f"Reduced {len(all_chunks)} chunks to {len(unique_chunks)} unique chunks")
        
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
        # logger.error(
        #     "Error retrieving document chunks | Document IDs: %s | Error: %s | Traceback: %s",
        #     documentIds,
        #     str(e),
        #     traceback.format_exc(),
        # )
        raise HTTPException(status_code=500, detail=str(e))





async def get_all_chunks_for_documents(document_ids: List[str], query: Optional[str] = None) -> List[Dict]:
    """
    Retrieve all chunks for given document_ids with metadata
    """
    chunks = []
    
    # logger.debug(f"Starting chunk retrieval for document_ids: {document_ids}")
    
    try:
        # For AsyncPgVector, we should use its methods properly
        if isinstance(vector_store, AsyncPgVector):
            # logger.info(f"Using AsyncPgVector query for document_ids: {document_ids}")
            
            # Query based only on file_id
            filter_dict = {
                "file_id": {"$in": document_ids}
            }
            
            # logger.debug(f"Using filter: {filter_dict}")
            
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
            
            # logger.debug(f"Vector search returned {len(all_docs)} documents")
            
            # Process results
            seen_contents = set()  # Track unique content
            for idx, (doc, score) in enumerate(all_docs):
                file_id = doc.metadata.get('file_id')
                
                # logger.debug(f"Processing document {idx}: file_id='{file_id}', score={score}")
                # if idx == 0:  # Log first document metadata for debugging
                #     logger.debug(f"Sample document metadata: {doc.metadata}")
                #     logger.debug(f"Content preview: {doc.page_content[:100]}...")
                
                # Skip duplicates based on content
                content_hash = get_content_hash(doc.page_content)
                if content_hash in seen_contents:
                    # logger.debug(f"Skipping duplicate content for file_id: {file_id}")
                    continue
                seen_contents.add(content_hash)
                
                if file_id in document_ids:
                    chunks.append({
                        "id": f"{file_id}_{idx}",
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "embedding": None
                    })
                    # logger.info(f"Found matching chunk for file_id: {file_id}, content length: {len(doc.page_content)}")
                else:
                    # logger.debug(f"file_id '{file_id}' not in target document_ids {document_ids}")
                    pass
                    
        else:
            # For other vector stores
            # logger.info(f"Using standard similarity search for document_ids: {document_ids}")
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
            
            # logger.debug(f"Standard search returned {len(all_docs)} documents")
            
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
                    # logger.info(f"Found matching chunk for file_id: {file_id}")
                    
    except Exception as e:
        # logger.error(f"Error in get_all_chunks_for_documents: {str(e)}\n{traceback.format_exc()}")
        raise
    
    if not chunks:
        # logger.warning(f"No chunks found for document_ids {document_ids}")
        # logger.warning("Check if the documents were properly embedded in the vector store")
        pass
    else:
        # logger.info(f"Found {len(chunks)} unique chunks for document_ids {document_ids}")
        pass
    
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
        
        # logger.info(f"Deduplication: {len(chunks)} -> {len(unique_chunks)} chunks")
        return unique_chunks
        
    except Exception as e:
        # logger.error(f"Error in deduplication: {str(e)}")
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
    
    NO CACHING - Fresh embeddings every time
    """
    
    # Fixed parameters
    MAX_CHUNKS = 3
    MIN_CHUNK_LENGTH = 50
    MAX_CHUNK_LENGTH = 1500
    
    # logger.info(f"=== PROJECT CHAT QUERY START ===")
    # logger.info(f"Query: '{query}'")
    # logger.info(f"Document IDs: {documentIds}")
    # logger.info(f"Document count: {len(documentIds)}")
    
    try:
        # 1. Handle empty query
        if not query or query.strip() == "":
            # logger.info("Empty query detected - using default search query")
            search_query = "document content"
        else:
            search_query = query
        
        # 2. Get query embedding (NO CACHING - fresh embedding every time)
        # logger.debug("Step 1: Getting fresh query embedding...")
        # logger.info(f"Getting fresh embedding for: '{search_query}'")
        
        # Try multiple methods to get fresh embedding
        query_embedding = None
        
        # Method 1: Try vector store's embedding service
        if hasattr(vector_store, 'embeddings') and vector_store.embeddings:
            # logger.info("Using vector store's embedding service")
            try:
                query_embedding = vector_store.embeddings.embed_query(search_query)
                # logger.info("Successfully got embedding from vector store")
            except Exception as e:
                # logger.error(f"Vector store embedding failed: {e}")
                pass
        
        # Method 2: Try from config
        if not query_embedding:
            # logger.info("Trying to get embedding service from config")
            try:
                from app.config import embeddings
                query_embedding = embeddings.embed_query(search_query)
                # logger.info("Successfully got embedding from config")
            except Exception as e:
                # logger.error(f"Config embedding failed: {e}")
                pass
        
        # Method 3: Direct OpenAI API call (fallback)
        if not query_embedding:
            # logger.info("Trying direct OpenAI API call for embedding")
            try:
                import openai
                import os
                
                # Get OpenAI client
                client = openai.AzureOpenAI(
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    api_version="2023-05-15",
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
                )
                
                # Get embedding directly
                response = client.embeddings.create(
                    input=search_query,
                    model="text-embedding-3-small"  # or your embedding model
                )
                
                query_embedding = response.data[0].embedding
                # logger.info("Successfully got embedding from direct OpenAI API")
            except Exception as e:
                # logger.error(f"Direct OpenAI embedding failed: {e}")
                pass
        
        # Final fallback: use the cached function if everything else fails
        if not query_embedding:
            # logger.warning("All embedding methods failed, falling back to cached function")
            from app.routes.document_routes import get_cached_query_embedding
            query_embedding = get_cached_query_embedding(search_query)
        
        # logger.info(f"Fresh query embedding obtained, length: {len(query_embedding) if query_embedding else 'None'}")
        
        # Add embedding hash for debugging
        import hashlib
        embedding_hash = hashlib.md5(str(query_embedding).encode()).hexdigest()
        # logger.info(f"Embedding hash: {embedding_hash}")
        
        # if query_embedding:
        #     logger.info(f"Embedding sample: {query_embedding[:5]}...{query_embedding[-5:]}")
        
        # 3. Perform similarity search with detailed logging
        # logger.debug("Step 2: Performing similarity search...")
        # logger.info(f"Vector store type: {type(vector_store)}")
        # logger.info(f"Using filter: file_id in {documentIds}")
        
        if isinstance(vector_store, AsyncPgVector):
            # logger.debug("Using AsyncPgVector similarity search")
            
            # DETAILED LOGGING: Check what's being passed to the vector store
            filter_dict = {"file_id": {"$in": documentIds}}
            # logger.info(f"=== VECTOR SEARCH PARAMETERS ===")
            # logger.info(f"search_query: '{search_query}'")
            # logger.info(f"query_embedding type: {type(query_embedding)}")
            # logger.info(f"query_embedding length: {len(query_embedding) if query_embedding else 'None'}")
            # logger.info(f"k parameter: {MAX_CHUNKS * 3}")
            # logger.info(f"filter_dict: {filter_dict}")
            # logger.info(f"filter_dict type: {type(filter_dict)}")
            
            # LOG: What method are we calling?
            # logger.info(f"Calling: vector_store.similarity_search_with_score_by_vector()")
            # logger.info(f"Method exists: {hasattr(vector_store, 'similarity_search_with_score_by_vector')}")
            
            # Try to get method signature info
            try:
                import inspect
                method = getattr(vector_store, 'similarity_search_with_score_by_vector')
                sig = inspect.signature(method)
                # logger.info(f"Method signature: {sig}")
            except Exception as e:
                # logger.info(f"Could not get method signature: {e}")
                pass
            
            # Make the call with detailed logging
            # logger.info("=== CALLING VECTOR STORE METHOD ===")
            documents_with_scores = await run_in_executor(
                None,
                vector_store.similarity_search_with_score_by_vector,
                query_embedding,
                k=MAX_CHUNKS * 3,
                filter=filter_dict
            )
            
            # logger.info(f"=== VECTOR SEARCH RESULTS ===")
            # logger.info(f"Initial search returned {len(documents_with_scores)} documents")
            
            # LOG: What did we actually get back?
            # logger.info(f"=== ANALYZING RETURNED DOCUMENTS ===")
            returned_file_ids = {}
            for idx, (doc, score) in enumerate(documents_with_scores):
                file_id = doc.metadata.get('file_id', 'MISSING')
                if file_id not in returned_file_ids:
                    returned_file_ids[file_id] = 0
                returned_file_ids[file_id] += 1
                
                # logger.info(f"Document {idx}:")
                # logger.info(f"  file_id: '{file_id}'")
                # logger.info(f"  score: {score}")
                # logger.info(f"  content_length: {len(doc.page_content)}")
                # logger.info(f"  content_start: {doc.page_content[:100]}...")
                # logger.info(f"  metadata: {doc.metadata}")
                # logger.info(f"  matches_filter: {file_id in documentIds}")
                
            # logger.info(f"=== SUMMARY OF RETURNED FILE_IDS ===")
            # logger.info(f"Requested: {documentIds}")
            # logger.info(f"Returned file_id counts: {returned_file_ids}")
            
            # Check if filter worked at all
            expected_files = set(documentIds)
            actual_files = set(returned_file_ids.keys())
            
            if expected_files.intersection(actual_files):
                # logger.info(f"✅ PARTIAL FILTER SUCCESS: Found {len(expected_files.intersection(actual_files))} expected files")
                pass
            else:
                # logger.error(f"❌ COMPLETE FILTER FAILURE: No expected files found")
                # logger.error(f"Expected: {expected_files}")
                # logger.error(f"Actual: {actual_files}")
                pass
                
            # If filter completely failed, let's try to understand why
            if not expected_files.intersection(actual_files):
                # logger.info("=== DEBUGGING FILTER FAILURE ===")
                
                # Try different filter formats
                # logger.info("Trying alternative filter format 1: direct equality")
                if len(documentIds) == 1:
                    alt_filter1 = {"file_id": documentIds[0]}
                    # logger.info(f"Alternative filter 1: {alt_filter1}")
                    
                    try:
                        alt_docs1 = await run_in_executor(
                            None,
                            vector_store.similarity_search_with_score_by_vector,
                            query_embedding,
                            k=10,
                            filter=alt_filter1
                        )
                        # logger.info(f"Alternative filter 1 returned: {len(alt_docs1)} documents")
                        
                        if len(alt_docs1) > 0:
                            # for idx, (doc, score) in enumerate(alt_docs1[:2]):
                            #     logger.info(f"Alt1 Doc {idx}: file_id='{doc.metadata.get('file_id')}', score={score}")
                            
                            # Use alternative results if they work
                            if alt_docs1[0][0].metadata.get('file_id') in documentIds:
                                # logger.info("Using alternative filter results")
                                documents_with_scores = alt_docs1
                        
                    except Exception as e:
                        # logger.error(f"Alternative filter 1 failed: {e}")
                        pass
                
                # Try no filter to see what's available
                # logger.info("Trying no filter to see available documents")
                try:
                    no_filter_docs = await run_in_executor(
                        None,
                        vector_store.similarity_search_with_score_by_vector,
                        query_embedding,
                        k=10
                    )
                    # logger.info(f"No filter returned: {len(no_filter_docs)} documents")
                    
                    available_file_ids = set()
                    for doc, score in no_filter_docs:
                        file_id = doc.metadata.get('file_id')
                        available_file_ids.add(file_id)
                    
                    # logger.info(f"Available file_ids in vector store: {list(available_file_ids)}")
                    # logger.info(f"Target file_id exists: {documentIds[0] in available_file_ids}")
                    
                except Exception as e:
                    # logger.error(f"No filter query failed: {e}")
                    pass
        
        else:
            # logger.debug("Using standard vector store similarity search")
            filter_dict = {"file_id": {"$in": documentIds}}
            
            documents_with_scores = vector_store.similarity_search_with_score_by_vector(
                query_embedding,
                k=MAX_CHUNKS * 3,
                filter=filter_dict
            )
        
        # logger.info(f"Final similarity search returned {len(documents_with_scores)} documents")
        
        # 4. Validate results - ensure we only get requested documents
        validated_documents = []
        for idx, (doc, score) in enumerate(documents_with_scores):
            file_id = doc.metadata.get('file_id')
            if file_id in documentIds:
                validated_documents.append((doc, score))
                # logger.info(f"✅ Validated document {idx}: file_id='{file_id}', score={score}")
            else:
                # logger.warning(f"❌ Filtered out document {idx}: file_id='{file_id}' not in {documentIds}")
                pass
        
        # logger.info(f"After validation: {len(validated_documents)} documents")
        
        if not validated_documents:
            # logger.error(f"❌ NO VALID DOCUMENTS FOUND for project documentIds: {documentIds}")
            # logger.error("This indicates the vector store filter is not working correctly")
            return []
        
        # 5. Log final selected documents
        # logger.info(f"=== FINAL VALIDATED DOCUMENTS ===")
        # for idx, (doc, score) in enumerate(validated_documents):
        #     logger.info(f"Final Doc {idx}:")
        #     logger.info(f"  file_id: '{doc.metadata.get('file_id')}'")
        #     logger.info(f"  score: {score}")
        #     logger.info(f"  content_length: {len(doc.page_content)}")
        #     logger.info(f"  content_preview: {doc.page_content[:200]}...")
        
        # 6. Process and score chunks
        # logger.debug("Step 3: Processing and scoring chunks...")
        
        chat_chunks = []
        seen_content_hashes = set()
        length_filtered_count = 0
        duplicate_filtered_count = 0
        
        for idx, (document, similarity_score) in enumerate(validated_documents):
            content = document.page_content
            metadata = document.metadata or {}
            file_id = metadata.get('file_id', 'unknown')
            
            # logger.debug(f"Processing chunk {idx}: file_id='{file_id}', content_length={len(content)}")
            
            # Filter by length
            if not (MIN_CHUNK_LENGTH <= len(content) <= MAX_CHUNK_LENGTH):
                length_filtered_count += 1
                # logger.debug(f"Chunk {idx} filtered by length: {len(content)} not in range [{MIN_CHUNK_LENGTH}, {MAX_CHUNK_LENGTH}]")
                continue
            
            # Deduplicate based on content
            content_hash = get_content_hash(content)
            if content_hash in seen_content_hashes:
                duplicate_filtered_count += 1
                # logger.debug(f"Chunk {idx} filtered as duplicate content")
                continue
            seen_content_hashes.add(content_hash)
            
            # Calculate chat relevance score
            chat_score = calculate_chat_relevance_score(
                content=content,
                query=query,  # Use original query for scoring
                similarity_score=similarity_score
            )
            
            # logger.debug(f"Chunk {idx} scored: relevance_score={chat_score}")
            
            chat_chunks.append({
                "content": content,
                "relevanceScore": chat_score,
                "similarity_score": similarity_score,
                "file_id": file_id
            })
        
        # logger.info(f"Chunk processing summary:")
        # logger.info(f"  Total retrieved: {len(validated_documents)}")
        # logger.info(f"  Length filtered: {length_filtered_count}")
        # logger.info(f"  Duplicate filtered: {duplicate_filtered_count}")
        # logger.info(f"  Final chunks: {len(chat_chunks)}")
        
        if not chat_chunks:
            # logger.warning("No chunks passed filtering! Check length and duplication filters.")
            return []
        
        # 7. Sort by relevance and select top chunks
        # logger.debug("Step 4: Sorting and selecting top chunks...")
        chat_chunks.sort(key=lambda x: x["relevanceScore"], reverse=True)
        selected_chunks = chat_chunks[:MAX_CHUNKS]
        
        # logger.info(f"=== FINAL SELECTED CHUNKS FOR LLM ===")
        # for idx, chunk in enumerate(selected_chunks):
        #     logger.info(f"Selected Chunk {idx}:")
        #     logger.info(f"  file_id: '{chunk['file_id']}'")
        #     logger.info(f"  relevance_score: {chunk['relevanceScore']}")
        #     logger.info(f"  similarity_score: {chunk['similarity_score']}")
        #     logger.info(f"  content_length: {len(chunk['content'])}")
        #     logger.info(f"  content_preview: {chunk['content'][:200]}...")
        #     logger.info(f"  content_end: ...{chunk['content'][-100:]}")
        
        # 8. Format response (just content strings)
        result = [chunk["content"] for chunk in selected_chunks]
        
        # Add final validation log
        # logger.info(f"=== FINAL CONTENT BEING RETURNED TO LLM ===")
        # for idx, content in enumerate(result):
        #     content_hash = hashlib.md5(content.encode()).hexdigest()
        #     logger.info(f"Result {idx}: hash={content_hash}, length={len(content)}")
        #     logger.info(f"  Content start: {content[:150]}...")
        #     logger.info(f"  Content end: ...{content[-150:]}")
        
        # logger.info(f"=== PROJECT CHAT QUERY SUCCESS: Returning {len(result)} content strings ===")
        return result
        
    except HTTPException:
        # logger.error("HTTPException in project_chat_query")
        raise
    except Exception as e:
        # logger.error(
        #     "Error in project chat query | Document IDs: %s | Query: %s | Error: %s | Traceback: %s",
        #     len(documentIds),
        #     query[:100] if query else "empty",
        #     str(e),
        #     traceback.format_exc(),
        # )
        raise HTTPException(status_code=500, detail=f"Project chat query failed: {str(e)}")


# Helper function to get content hash for deduplication
def get_content_hash(content: str) -> str:
    """Generate a hash of normalized content for deduplication"""
    import hashlib
    # Normalize whitespace and case for better matching
    normalized = " ".join(content.lower().split())
    return hashlib.md5(normalized.encode()).hexdigest()


# Helper function to calculate chat relevance score
def calculate_chat_relevance_score(content: str, query: str, similarity_score: float) -> float:
    """
    Calculate relevance score specifically for chat responses.
    Combines semantic similarity with chat-specific factors.
    """
    import re
    
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