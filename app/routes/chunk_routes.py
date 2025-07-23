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

async def get_enhanced_chunks_with_expansion(
    document_ids: List[str], 
    query: str, 
    max_chunks: int,
    use_faq_scoring: bool = True
) -> List[ChunkResponse]:
    """
    Enhanced chunk retrieval with smart query expansion.
    Only expands queries with less than 3 words.
    """
    
    logger.info(f"🔍 ENHANCED CHUNK RETRIEVAL WITH EXPANSION:")
    logger.info(f"  Original query: '{query}'")
    
    # Step 1: Try the original query
    primary_chunks = await process_document_chunks(
        document_ids=document_ids,
        query=query,
        max_chunks=max_chunks,
        min_chunk_length=FAQ_MIN_CHUNK_LENGTH if use_faq_scoring else CHAT_MIN_CHUNK_LENGTH,
        max_chunk_length=FAQ_MAX_CHUNK_LENGTH if use_faq_scoring else CHAT_MAX_CHUNK_LENGTH,
        diversity_weight=FAQ_DIVERSITY_WEIGHT if use_faq_scoring else CHAT_DIVERSITY_WEIGHT,
        similarity_threshold=(FAQ_SIMILARITY_THRESHOLD * 1.3) if use_faq_scoring else (CHAT_SIMILARITY_THRESHOLD * 1.3),
        relevance_threshold=(FAQ_RELEVANCE_THRESHOLD * 0.7) if use_faq_scoring else (CHAT_RELEVANCE_THRESHOLD * 0.7),
        keyword_overlap_threshold=(FAQ_KEYWORD_OVERLAP_THRESHOLD * 0.5) if use_faq_scoring else (CHAT_KEYWORD_OVERLAP_THRESHOLD * 0.5),
        use_faq_scoring=use_faq_scoring
    )
    
    logger.info(f"  Primary query returned: {len(primary_chunks)} chunks")
    
    # Step 2: Check if query expansion is needed
    query_words = query.strip().split()
    should_expand = len(query_words) < 3
    
    logger.info(f"  Query has {len(query_words)} words - {'will expand' if should_expand else 'no expansion needed'}")
    
    if not should_expand:
        logger.info(f"  Using primary results only: {len(primary_chunks)} chunks")
        return primary_chunks
    
    # Step 3: Only expand short queries (< 3 words)
    target_chunk_count = max_chunks // 2
    
    if len(primary_chunks) < target_chunk_count:
        logger.info(f"  Expanding short query - need {target_chunk_count}, got {len(primary_chunks)}")
        
        # Generate expanded queries for short queries only
        expanded_queries = generate_query_expansions(query)
        
        additional_chunks = []
        chunks_needed = max_chunks - len(primary_chunks)
        chunks_per_expansion = max(1, chunks_needed // len(expanded_queries))
        
        for expanded_query in expanded_queries:
            logger.info(f"  Trying expanded query: '{expanded_query}'")
            
            extra_chunks = await process_document_chunks(
                document_ids=document_ids,
                query=expanded_query,
                max_chunks=chunks_per_expansion,
                min_chunk_length=FAQ_MIN_CHUNK_LENGTH if use_faq_scoring else CHAT_MIN_CHUNK_LENGTH,
                max_chunk_length=FAQ_MAX_CHUNK_LENGTH if use_faq_scoring else CHAT_MAX_CHUNK_LENGTH,
                diversity_weight=FAQ_DIVERSITY_WEIGHT if use_faq_scoring else CHAT_DIVERSITY_WEIGHT,
                similarity_threshold=(FAQ_SIMILARITY_THRESHOLD * 1.5) if use_faq_scoring else (CHAT_SIMILARITY_THRESHOLD * 1.5),
                relevance_threshold=(FAQ_RELEVANCE_THRESHOLD * 0.5) if use_faq_scoring else (CHAT_RELEVANCE_THRESHOLD * 0.5),
                keyword_overlap_threshold=(FAQ_KEYWORD_OVERLAP_THRESHOLD * 0.3) if use_faq_scoring else (CHAT_KEYWORD_OVERLAP_THRESHOLD * 0.3),
                use_faq_scoring=use_faq_scoring
            )
            
            logger.info(f"    Expanded query '{expanded_query}' returned: {len(extra_chunks)} chunks")
            additional_chunks.extend(extra_chunks)
            
            # Stop if we have enough chunks
            if len(primary_chunks) + len(additional_chunks) >= max_chunks:
                break
        
        # Combine and deduplicate all chunks
        all_chunks = primary_chunks + additional_chunks
        
        # Convert to dict format for deduplication
        chunks_for_dedup = []
        for chunk in all_chunks:
            chunks_for_dedup.append({
                "id": chunk.chunkId,
                "content": chunk.content,
                "metadata": chunk.metadata,
                "faq_score": chunk.score if use_faq_scoring else chunk.characteristics.get("final_score", chunk.score),
                "characteristics": chunk.characteristics
            })
        
        # Deduplicate
        unique_chunks = deduplicate_chunks(chunks_for_dedup, similarity_threshold=0.85)
        
        # Convert back to ChunkResponse and sort by score
        final_chunks = []
        for chunk in unique_chunks:
            final_chunk = ChunkResponse(
                chunkId=chunk["id"],
                content=chunk["content"],
                metadata=chunk["metadata"],
                score=chunk["faq_score"],
                characteristics=chunk.get("characteristics", {})
            )
            final_chunks.append(final_chunk)
        
        # Sort by score and limit to max_chunks
        final_chunks.sort(key=lambda x: x.score, reverse=True)
        final_chunks = final_chunks[:max_chunks]
        
        logger.info(f"  Final result: {len(final_chunks)} chunks after expansion and deduplication")
        return final_chunks
    
    logger.info(f"  Using primary results: {len(primary_chunks)} chunks (sufficient for short query)")
    return primary_chunks


def generate_query_expansions(query: str) -> List[str]:
    """
    Generate 3 high-quality expanded search queries based on the original query.
    Completely generic - works for any domain without assumptions.
    """
    import re
    
    query_lower = query.lower().strip()
    
    # Generate exactly 3 generic expansions
    expansions = [
        f"what are {query_lower}",      # Definition/explanation
        f"how do {query_lower} work",   # Process/functionality
        f"about {query_lower}"          # General information
    ]
    
    logger.info(f"  Generated 3 query expansions: {expansions}")
    return expansions

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
    
    logger.info(f"🚀 ===== DOCUMENT CHUNK PROCESSING START =====")
    logger.info(f"  🔍 Query: '{search_query}'")
    logger.info(f"  📁 Document IDs: {document_ids}")
    logger.info(f"  ⚙️  Parameters:")
    logger.info(f"    📊 Max chunks: {max_chunks}")
    logger.info(f"    📏 Length range: {min_chunk_length}-{max_chunk_length}")
    logger.info(f"    🎯 Similarity threshold: {similarity_threshold}")
    logger.info(f"    🎲 Use FAQ scoring: {use_faq_scoring}")
    if not use_faq_scoring:
        logger.info(f"    📈 Relevance threshold: {relevance_threshold}")
        logger.info(f"    🔤 Keyword overlap threshold: {keyword_overlap_threshold}")
    
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
    
    # Try multiple filter formats for different vector stores
    filter_formats = [
        {"file_id": {"$in": document_ids}},  # MongoDB style
        {"file_id": document_ids[0]} if len(document_ids) == 1 else {"file_id": {"$in": document_ids}},  # Single document
        {"metadata.file_id": {"$in": document_ids}},  # Alternative metadata format
        {"metadata.file_id": document_ids[0]} if len(document_ids) == 1 else {"metadata.file_id": {"$in": document_ids}},
    ]
    
    logger.info(f"🔍 SIMILARITY SEARCH:")
    logger.info(f"  📁 Target document IDs: {document_ids}")
    logger.info(f"  📊 Max candidates: {max_candidates}")
    
    documents_with_scores = None
    successful_filter = None
    
    # Try different filter formats until one works
    for i, filter_dict in enumerate(filter_formats):
        try:
            logger.info(f"  🔧 Trying filter format {i+1}: {filter_dict}")
            
            if isinstance(vector_store, AsyncPgVector):
                test_docs = await run_in_executor(
                    None,
                    vector_store.similarity_search_with_score_by_vector,
                    query_embedding,
                    max_candidates,
                    filter_dict
                )
            else:
                test_docs = vector_store.similarity_search_with_score_by_vector(
                    query_embedding,
                    k=max_candidates,
                    filter=filter_dict
                )
            
            # Check if filter worked by examining returned file IDs
            if test_docs:
                returned_file_ids = set()
                for doc, score in test_docs:
                    file_id = doc.metadata.get('file_id')
                    returned_file_ids.add(file_id)
                
                # Check if filter worked (only returned docs from target IDs)
                target_id_set = set(document_ids)
                if returned_file_ids.issubset(target_id_set):
                    logger.info(f"  ✅ Filter format {i+1} WORKED! Got {len(test_docs)} docs from target IDs only")
                    documents_with_scores = test_docs
                    successful_filter = filter_dict
                    break
                else:
                    logger.warning(f"  ❌ Filter format {i+1} failed - got docs from: {returned_file_ids}")
            else:
                logger.warning(f"  ❌ Filter format {i+1} returned no documents")
                
        except Exception as e:
            logger.warning(f"  ❌ Filter format {i+1} caused error: {str(e)}")
            continue
    
    # If no filter worked, do unfiltered search and filter manually (last resort)
    if documents_with_scores is None:
        logger.error(f"  ❌ ALL FILTER FORMATS FAILED! Falling back to manual filtering")
        logger.error(f"  ⚠️  This is VERY INEFFICIENT - your vector store filter is broken!")
        
        try:
            if isinstance(vector_store, AsyncPgVector):
                all_docs = await run_in_executor(
                    None,
                    vector_store.similarity_search_with_score_by_vector,
                    query_embedding,
                    max_candidates * 10,  # Get more since we'll filter manually
                    None  # No filter
                )
            else:
                all_docs = vector_store.similarity_search_with_score_by_vector(
                    query_embedding,
                    k=max_candidates * 10,
                    filter=None
                )
            
            # Manually filter to target documents
            documents_with_scores = [
                (doc, score) for doc, score in all_docs
                if doc.metadata.get('file_id') in document_ids
            ][:max_candidates]  # Limit to requested count
            
            logger.warning(f"  ⚠️  Manual filtering kept {len(documents_with_scores)} docs from {len(all_docs)} total")
            
        except Exception as e:
            logger.error(f"  ❌ Even unfiltered search failed: {str(e)}")
            return []
    
    if documents_with_scores:
        logger.info(f"  📥 Final result: {len(documents_with_scores)} documents")
        returned_file_ids = set(doc.metadata.get('file_id') for doc, _ in documents_with_scores)
        logger.info(f"  🆔 File IDs in results: {list(returned_file_ids)}")
        
        if successful_filter:
            logger.info(f"  ✅ Successful filter: {successful_filter}")
    
    return documents_with_scores or []


def filter_by_similarity_threshold(documents_with_scores, document_ids: List[str], similarity_threshold: float):
    """Filter documents by similarity threshold."""
    validated_documents = []
    similarity_filtered_count = 0
    
    logger.info(f"🔍 SIMILARITY THRESHOLD FILTERING:")
    logger.info(f"  📊 Threshold: {similarity_threshold}")
    logger.info(f"  📄 Target document IDs: {document_ids}")
    logger.info(f"  📥 Input documents: {len(documents_with_scores)}")
    
    for idx, (doc, similarity_score) in enumerate(documents_with_scores):
        file_id = doc.metadata.get('file_id')
        
        logger.info(f"\n  📄 Document {idx+1}/{len(documents_with_scores)}:")
        logger.info(f"    🆔 File ID: '{file_id}'")
        logger.info(f"    📊 Similarity score: {similarity_score}")
        logger.info(f"    📏 Content length: {len(doc.page_content)} chars")
        logger.info(f"    📝 Content preview: '{doc.page_content[:150]}...'")
        logger.info(f"    🗂️  Metadata: {doc.metadata}")
        
        # Check if document belongs to project
        if file_id not in document_ids:
            logger.info(f"    ❌ File ID not in target documents")
            continue
            
        logger.info(f"    ✅ File ID matches target documents")
        
        # Apply similarity threshold
        if similarity_score > similarity_threshold:
            similarity_filtered_count += 1
            logger.info(f"    ❌ Similarity filter: {similarity_score} > threshold {similarity_threshold}")
            continue
            
        logger.info(f"    ✅ Similarity OK: {similarity_score} <= threshold {similarity_threshold}")
        logger.info(f"    🎉 DOCUMENT ACCEPTED")
        
        validated_documents.append((doc, similarity_score))
    
    logger.info(f"\n📊 SIMILARITY FILTERING SUMMARY:")
    logger.info(f"  📥 Input documents: {len(documents_with_scores)}")
    logger.info(f"  ❌ Similarity filtered: {similarity_filtered_count}")
    logger.info(f"  ✅ Passed documents: {len(validated_documents)}")
    
    if validated_documents:
        scores = [score for _, score in validated_documents]
        logger.info(f"  📈 Score range: {min(scores):.4f} - {max(scores):.4f}")
    
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
    
    logger.info(f"🔄 PROCESSING {len(validated_documents)} CHUNKS FOR CHAT:")
    logger.info(f"  📊 Thresholds: relevance={relevance_threshold}, keyword_overlap={keyword_overlap_threshold}")
    logger.info(f"  📏 Length limits: {min_chunk_length}-{max_chunk_length} chars")
    
    for idx, (document, similarity_score) in enumerate(validated_documents):
        content = document.page_content
        metadata = document.metadata or {}
        file_id = metadata.get('file_id', 'unknown')
        
        logger.info(f"\n  📄 CHUNK {idx+1}/{len(validated_documents)} (file: {file_id}):")
        
        # Filter by length
        if not (min_chunk_length <= len(content) <= max_chunk_length):
            length_filtered_count += 1
            logger.info(f"    ❌ Length filter: {len(content)} chars not in range [{min_chunk_length}, {max_chunk_length}]")
            continue
        
        logger.info(f"    ✅ Length OK: {len(content)} chars")
        
        # Deduplicate based on content
        content_hash = get_content_hash(content)
        if content_hash in seen_content_hashes:
            duplicate_filtered_count += 1
            logger.info(f"    ❌ Duplicate filter: content hash {content_hash[:8]}...")
            continue
        seen_content_hashes.add(content_hash)
        logger.info(f"    ✅ Unique content: hash {content_hash[:8]}...")
        
        # Calculate enhanced relevance score
        relevance_metrics = calculate_enhanced_chat_relevance_score(
            content=content,
            query=query,
            similarity_score=similarity_score
        )
        
        # Apply relevance threshold
        if relevance_metrics["final_score"] < relevance_threshold:
            relevance_filtered_count += 1
            logger.info(f"    ❌ Relevance filter: {relevance_metrics['final_score']:.4f} < {relevance_threshold}")
            continue
        
        logger.info(f"    ✅ Relevance OK: {relevance_metrics['final_score']:.4f} >= {relevance_threshold}")
        
        # Apply keyword overlap threshold
        if relevance_metrics["keyword_overlap"] < keyword_overlap_threshold:
            keyword_filtered_count += 1
            logger.info(f"    ❌ Keyword filter: {relevance_metrics['keyword_overlap']:.4f} < {keyword_overlap_threshold}")
            continue
        
        logger.info(f"    ✅ Keyword overlap OK: {relevance_metrics['keyword_overlap']:.4f} >= {keyword_overlap_threshold}")
        logger.info(f"    🎉 CHUNK ACCEPTED with score: {relevance_metrics['final_score']:.4f}")
        
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
    
    logger.info(f"\n📊 CHAT CHUNK FILTERING SUMMARY:")
    logger.info(f"  📥 Total retrieved: {len(validated_documents)}")
    logger.info(f"  📏 Length filtered: {length_filtered_count}")
    logger.info(f"  🔄 Duplicate filtered: {duplicate_filtered_count}")
    logger.info(f"  🎯 Relevance filtered: {relevance_filtered_count}")
    logger.info(f"  🔤 Keyword filtered: {keyword_filtered_count}")
    logger.info(f"  ✅ Final chunks: {len(chat_chunks)}")
    
    # if chat_chunks:
    #     logger.info(f"  🏆 Top scores: {[f'{chunk["relevanceScore"]:.3f}' for chunk in sorted(chat_chunks, key=lambda x: x['relevanceScore'], reverse=True)[:3]]}")
    
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
    query: Optional[str] = Query(None, description="Optional query to filter chunks"),
    useExpansion: bool = Query(True, description="Use query expansion for better results")
):
    """
    Extract the best document chunks using enhanced retrieval with query expansion.
    Returns List[str] of content strings optimized for FAQ generation.
    """
    
    try:
        # Use enhanced retrieval with expansion if enabled
        if useExpansion and query and query.strip():
            chunk_responses = await get_enhanced_chunks_with_expansion(
                document_ids=documentIds,
                query=query,
                max_chunks=maxChunks,
                use_faq_scoring=True
            )
        else:
            # Fall back to original method
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
        
        logger.info(f"📊 FAQ CHUNK RETRIEVAL SUCCESS:")
        logger.info(f"  Query: '{query}'")
        logger.info(f"  Chunks returned: {len(content_strings)}")
        logger.info(f"  Expansion used: {useExpansion}")
        
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
    Enhanced generic relevance scoring that prioritizes exact query term matches
    and related concepts without hardcoded domain knowledge.
    """
    import re
    
    metrics = {
        "semantic_score": 0.0,
        "exact_query_match": 0.0,        # NEW: Direct query term matching
        "keyword_overlap": 0.0,
        "contextual_relevance": 0.0,
        "query_term_density": 0.0,       # NEW: How often query terms appear
        "related_term_boost": 0.0,       # NEW: Terms that appear near query terms
        "exact_phrase_match": 0.0,
        "qa_indicators": 0.0,
        "completeness": 0.0,
        "length_penalty": 0.0,
        "final_score": 0.0
    }
    
    query_lower = query.lower().strip()
    content_lower = content.lower()
    
    # 1. Semantic similarity score
    if similarity_score <= 1.0:
        metrics["semantic_score"] = max(0, 1.0 - similarity_score)
    else:
        metrics["semantic_score"] = 1.0 / (1.0 + similarity_score)
    
    # 2. NEW: Exact query term matching with high weight
    query_terms = [term.strip() for term in re.findall(r'\b\w+\b', query_lower) if len(term.strip()) > 2]
    if query_terms:
        exact_matches = 0
        for term in query_terms:
            # Count exact word boundary matches
            pattern = r'\b' + re.escape(term) + r'\b'
            matches = len(re.findall(pattern, content_lower))
            exact_matches += min(matches, 3)  # Cap at 3 per term to avoid over-weighting
        
        metrics["exact_query_match"] = min(exact_matches / (len(query_terms) * 2), 1.0)
        logger.info(f"  🎯 Exact query matches: {exact_matches} for terms {query_terms} = {metrics['exact_query_match']:.4f}")
    
    # 3. Enhanced keyword overlap
    def extract_meaningful_words(text: str) -> set:
        clean_text = re.sub(r'[^\w\s]', ' ', text)
        words = clean_text.split()
        meaningful_words = set()
        for word in words:
            word_lower = word.lower()
            if (len(word_lower) > 2 or 
                re.match(r'^\d+$', word_lower) or 
                word_lower in {'is', 'it', 'we', 'do', 'ai', 'qa', 'ui', 'ux', 'id', 'us', 'or'}):
                meaningful_words.add(word_lower)
        return meaningful_words
    
    query_words = extract_meaningful_words(query_lower)
    content_words = extract_meaningful_words(content_lower)
    
    if query_words:
        overlapping_words = query_words.intersection(content_words)
        metrics["keyword_overlap"] = len(overlapping_words) / len(query_words)
    
    # 4. NEW: Query term density (how concentrated are query terms in the content)
    if query_terms:
        total_words = len(content_lower.split())
        if total_words > 0:
            query_word_count = sum(len(re.findall(r'\b' + re.escape(term) + r'\b', content_lower)) for term in query_terms)
            metrics["query_term_density"] = min(query_word_count / total_words * 10, 1.0)  # Scale up for visibility
    
    # 5. NEW: Related term boost (terms that appear near query terms)
    def find_related_terms_boost(content: str, query_terms: List[str], window_size: int = 20) -> float:
        """Find terms that appear within a window around query terms"""
        if not query_terms:
            return 0.0
        
        words = content_lower.split()
        related_boost = 0.0
        
        for i, word in enumerate(words):
            if word in query_terms:
                # Look at surrounding words
                start = max(0, i - window_size)
                end = min(len(words), i + window_size + 1)
                context_words = words[start:end]
                
                # Boost for financial terms, feature terms, etc. near query terms
                boost_patterns = [
                    r'\$[\d,]+',  # Prices
                    r'\d+[kmb]',  # Numbers with K/M/B
                    r'\d+\.\d+',  # Decimals
                    r'per\s+\w+', # "per month", "per project"
                    r'includes?', r'features?', r'benefits?',
                    r'unlimited', r'limited', r'additional',
                    r'upgrade', r'expand', r'increase'
                ]
                
                for context_word in context_words:
                    for pattern in boost_patterns:
                        if re.search(pattern, context_word):
                            related_boost += 0.1
        
        return min(related_boost, 1.0)
    
    if query_terms:
        metrics["related_term_boost"] = find_related_terms_boost(content_lower, query_terms)
    
    # 6. Contextual relevance (existing logic)
    metrics["contextual_relevance"] = calculate_contextual_relevance(query_lower, content)
    
    # 7. Exact phrase matching (existing logic)
    query_phrases = []
    if len(query.split()) > 1:
        words = re.sub(r'[^\w\s]', ' ', query_lower).split()
        for i in range(len(words) - 1):
            if i + 2 <= len(words):
                phrase = ' '.join(words[i:i+2])
                if phrase.strip():
                    query_phrases.append(phrase)
    
    content_clean = re.sub(r'[^\w\s]', ' ', content_lower)
    exact_matches = sum(1 for phrase in query_phrases if phrase in content_clean)
    metrics["exact_phrase_match"] = min(exact_matches / max(len(query_phrases), 1), 1.0) if query_phrases else 0.0
    
    # 8. QA indicators (existing logic)
    qa_patterns = [
        r'\b(what|why|how|when|where|who|which)\b',
        r'\b(is|are|can|should|must|will|does|do)\b',
        r'\b(definition|meaning|purpose|explanation|description)\b',
        r'\b(example|instance|such as|like|including)\b',
        r'\b(because|therefore|thus|hence|so|since)\b',
        r'\b(first|second|third|finally|then|next|step)\b',
        r'\b(important|key|main|primary|essential|significant)\b'
    ]
    
    qa_score = sum(1 for pattern in qa_patterns if re.search(pattern, content_lower))
    metrics["qa_indicators"] = min(qa_score / len(qa_patterns), 1.0)
    
    # 9. Completeness (existing logic)
    completeness_indicators = [
        r'[.!?]\s+[A-Z]',
        r'\b(however|but|although|while|moreover|furthermore)\b',
        r':\s*\n',
        r'^\s*\d+[.)]\s',
        r'^\s*[•\-*]\s',
        r'\n\s*\n'
    ]
    
    completeness_score = sum(1 for pattern in completeness_indicators if re.search(pattern, content, re.MULTILINE))
    metrics["completeness"] = min(completeness_score / len(completeness_indicators), 1.0)
    
    # 10. Length penalty (existing logic)
    optimal_length = 600
    length_diff = abs(len(content) - optimal_length)
    metrics["length_penalty"] = max(0.4, 1.0 - (length_diff / optimal_length))
    
    # 11. UPDATED WEIGHTS: Prioritize exact matches for specific queries
    is_specific_term_query = len(query_terms) <= 3 and any(len(term) > 3 for term in query_terms)
    
    if is_specific_term_query:
        # For specific term queries like "add-ons", prioritize exact matches
        weights = {
            "semantic_score": 0.15,
            "exact_query_match": 0.35,      # High weight for exact matches
            "keyword_overlap": 0.20,
            "contextual_relevance": 0.10,
            "query_term_density": 0.10,     # Boost content with multiple mentions
            "related_term_boost": 0.05,     # Boost for related terms
            "exact_phrase_match": 0.03,
            "qa_indicators": 0.01,
            "completeness": 0.01,
            "length_penalty": 0.00
        }
        logger.info(f"  🎯 Using SPECIFIC TERM weights for query: {query}")
    else:
        # For general queries, use balanced approach
        weights = {
            "semantic_score": 0.25,
            "exact_query_match": 0.20,
            "keyword_overlap": 0.25,
            "contextual_relevance": 0.15,
            "query_term_density": 0.05,
            "related_term_boost": 0.05,
            "exact_phrase_match": 0.03,
            "qa_indicators": 0.01,
            "completeness": 0.01,
            "length_penalty": 0.00
        }
        logger.info(f"  🎯 Using BALANCED weights")
    
    # Calculate final score
    metrics["final_score"] = sum(metrics[key] * weights[key] for key in weights.keys())
    
    # Log detailed scoring
    logger.info(f"  📊 Scoring breakdown:")
    for key, weight in weights.items():
        if weight > 0:
            component_score = metrics[key] * weight
            logger.info(f"    {key}: {metrics[key]:.4f} × {weight:.2f} = {component_score:.4f}")
    
    logger.info(f"  🎯 FINAL SCORE: {metrics['final_score']:.4f}")
    
    # Round scores
    for key in metrics:
        metrics[key] = round(metrics[key], 4)
    
    return metrics


def calculate_contextual_relevance(query: str, content: str) -> float:
    """Existing contextual relevance function - keeping unchanged"""
    score = 0.0
    query_lower = query.lower()
    content_lower = content.lower()
    
    query_patterns = {
        'question_words': re.findall(r'\b(what|how|why|when|where|who|which|is|are|can|do|does|will|would|should)\b', query_lower),
        'action_words': re.findall(r'\b(get|find|need|want|help|show|tell|explain|describe)\b', query_lower),
        'quantity_words': re.findall(r'\b(much|many|often|long|far|big|small|fast|slow)\b', query_lower),
        'comparison_words': re.findall(r'\b(better|best|worse|different|same|compare|versus|vs)\b', query_lower),
    }
    
    content_patterns = {
        'headings': len(re.findall(r'^[A-Z][^.!?]*', content, re.MULTILINE)),
        'lists': len(re.findall(r'^\s*[-•*]\s', content, re.MULTILINE)),
        'numbers': len(re.findall(r'\b\d+\b', content)),
        'definitions': len(re.findall(r'\b(is|are|means|refers to|defined as|called)\b', content_lower)),
        'procedures': len(re.findall(r'\b(step|first|then|next|finally|process|procedure)\b', content_lower)),
        'explanations': len(re.findall(r'\b(because|since|therefore|thus|however|although)\b', content_lower)),
    }
    
    alignment_scores = []
    
    if query_patterns['question_words']:
        if content_patterns['definitions'] > 0 or content_patterns['explanations'] > 0:
            alignment_scores.append(0.8)
        elif content_patterns['headings'] > 0:
            alignment_scores.append(0.6)
        else:
            alignment_scores.append(0.3)
    
    if any(word in query_lower for word in ['how', 'steps', 'process', 'setup', 'install']):
        if content_patterns['procedures'] > 2:
            alignment_scores.append(0.9)
        elif content_patterns['lists'] > 0:
            alignment_scores.append(0.7)
        else:
            alignment_scores.append(0.2)
    
    if query_patterns['quantity_words'] or query_patterns['comparison_words']:
        if content_patterns['numbers'] > 3:
            alignment_scores.append(0.8)
        elif content_patterns['lists'] > 1:
            alignment_scores.append(0.6)
        else:
            alignment_scores.append(0.2)
    
    info_density = min((content_patterns['headings'] + content_patterns['numbers'] + 
                       content_patterns['definitions']) / 10, 1.0)
    alignment_scores.append(info_density)
    
    if alignment_scores:
        score = sum(alignment_scores) / len(alignment_scores)
    else:
        score = 0.5
    
    return score