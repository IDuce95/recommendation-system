from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Optional
from pydantic import BaseModel
import logging

from ai.rag_node import create_rag_node, RAGNode
from ai.knowledge_base_generator import setup_fake_knowledge_base

router = APIRouter(prefix="/rag", tags=["RAG"])
logger = logging.getLogger(__name__)

rag_node_instance: Optional[RAGNode] = None

class RAGQuery(BaseModel):
    query: str
    user_id: Optional[str] = None
    n_results: int = 5

class RAGResponse(BaseModel):
    response: str
    contexts: List[Dict]
    metadata: Dict

class KnowledgeBaseSetupResponse(BaseModel):
    success: bool
    message: str
    statistics: Optional[Dict] = None

def get_rag_node() -> RAGNode:
    global rag_node_instance
    if rag_node_instance is None:
        try:
            rag_node_instance = create_rag_node()
        except Exception as e:
            logger.error(f"Failed to create RAG node: {e}")
            raise HTTPException(status_code=500, detail="RAG service unavailable")
    return rag_node_instance

@router.post("/query", response_model=RAGResponse)
async def rag_query(query_data: RAGQuery, rag_node: RAGNode = Depends(get_rag_node)) -> RAGResponse:
    try:
        contexts = rag_node.retrieve_relevant_context(
            query_data.query,
            user_id=query_data.user_id,
            n_results=query_data.n_results
        )

        user_profile = ""
        if query_data.user_id:
            user_profile = f"User ID: {query_data.user_id}"

        response = rag_node.generate_rag_response(query_data.query, contexts, user_profile)

        metadata = {
            "query": query_data.query,
            "user_id": query_data.user_id,
            "contexts_found": len(contexts),
            "sources": list(set([ctx['source'] for ctx in contexts])) if contexts else []
        }

        return RAGResponse(
            response=response,
            contexts=contexts,
            metadata=metadata
        )

    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")

@router.get("/similar-products/{product_id}")
async def get_similar_products(
    product_id: str,
    n_results: int = 5,
    rag_node: RAGNode = Depends(get_rag_node)
) -> Dict:
    try:
        similar_products = rag_node.get_similar_products(product_id, n_results=n_results)

        return {
            "product_id": product_id,
            "similar_products": similar_products,
            "count": len(similar_products)
        }

    except Exception as e:
        logger.error(f"Similar products query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Similar products query failed: {str(e)}")

@router.post("/setup-knowledge-base", response_model=KnowledgeBaseSetupResponse)
async def setup_knowledge_base() -> KnowledgeBaseSetupResponse:
    try:
        result = setup_fake_knowledge_base()

        if result:
            return KnowledgeBaseSetupResponse(
                success=True,
                message="Knowledge base setup completed successfully",
                statistics=result
            )
        else:
            return KnowledgeBaseSetupResponse(
                success=False,
                message="Knowledge base setup failed"
            )

    except Exception as e:
        logger.error(f"Knowledge base setup failed: {e}")
        return KnowledgeBaseSetupResponse(
            success=False,
            message=f"Knowledge base setup failed: {str(e)}"
        )

@router.get("/collections")
async def list_collections(rag_node: RAGNode = Depends(get_rag_node)) -> Dict:
    try:
        collections = rag_node.chroma_client.list_collections()

        collection_info = {}
        for collection_name in collections:
            info = rag_node.chroma_client.get_collection_info(collection_name)
            collection_info[collection_name] = info

        return {
            "collections": collection_info,
            "total_collections": len(collections)
        }

    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")

@router.post("/update-user-context")
async def update_user_context(
    user_id: str,
    interaction_data: Dict,
    rag_node: RAGNode = Depends(get_rag_node)
) -> Dict:
    try:
        rag_node.update_user_context(user_id, interaction_data)

        return {
            "success": True,
            "message": f"User context updated for {user_id}",
            "interaction_data": interaction_data
        }

    except Exception as e:
        logger.error(f"Failed to update user context: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update user context: {str(e)}")

@router.get("/health")
async def rag_health_check() -> Dict:
    try:
        rag_node = get_rag_node()
        collections = rag_node.chroma_client.list_collections()

        return {
            "status": "healthy",
            "rag_available": True,
            "chromadb_connected": True,
            "collections_count": len(collections),
            "collections": collections
        }

    except Exception as e:
        logger.error(f"RAG health check failed: {e}")
        return {
            "status": "unhealthy",
            "rag_available": False,
            "chromadb_connected": False,
            "error": str(e)
        }
