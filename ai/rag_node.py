from typing import Dict, List, Any, Optional
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
import json
from ai.chromadb_client import ChromaDBClient, KnowledgeBaseManager
from ai.graph_state import GraphState
from config.prompt_config import PromptConfig

class RAGNode:
    def __init__(self, chroma_client: ChromaDBClient, llm: Optional[LLM] = None):
        self.chroma_client = chroma_client
        self.kb_manager = KnowledgeBaseManager(chroma_client)
        self.llm = llm
        self.prompt_config = PromptConfig()
        self.prompts = self.prompt_config.get_all_prompts()

        self.rag_prompt = PromptTemplate(
            input_variables=["context", "query", "user_profile"],
            template=self.prompts.get("rag_template", """
            Based on the following context and user profile, provide a helpful recommendation response.

            Context: {context}
            User Profile: {user_profile}
            Query: {query}

            Response:
            """)
        )

    def retrieve_relevant_context(self, query: str, user_id: Optional[str] = None, n_results: int = 5) -> List[Dict]:
        contexts = []

        try:
            product_results = self.kb_manager.search_product_knowledge(query, n_results=n_results)

            for i, doc in enumerate(product_results.get('documents', [[]])[0]):
                if i < len(product_results.get('metadatas', [[]])[0]):
                    metadata = product_results['metadatas'][0][i]
                    contexts.append({
                        "content": doc,
                        "metadata": metadata,
                        "source": "product_knowledge",
                        "relevance_score": 1.0 - (i * 0.1)
                    })
        except Exception as e:
            print(f"Error retrieving product knowledge: {e}")

        if user_id:
            try:
                user_results = self.kb_manager.search_user_knowledge(query, user_id=user_id, n_results=3)

                for i, doc in enumerate(user_results.get('documents', [[]])[0]):
                    if i < len(user_results.get('metadatas', [[]])[0]):
                        metadata = user_results['metadatas'][0][i]
                        contexts.append({
                            "content": doc,
                            "metadata": metadata,
                            "source": "user_knowledge",
                            "relevance_score": 0.9 - (i * 0.1)
                        })
            except Exception as e:
                print(f"Error retrieving user knowledge: {e}")

        try:
            snippet_results = self.chroma_client.query_collection("knowledge_snippets", [query], n_results=3)

            for i, doc in enumerate(snippet_results.get('documents', [[]])[0]):
                if i < len(snippet_results.get('metadatas', [[]])[0]):
                    metadata = snippet_results['metadatas'][0][i]
                    contexts.append({
                        "content": doc,
                        "metadata": metadata,
                        "source": "knowledge_snippets",
                        "relevance_score": 0.8 - (i * 0.1)
                    })
        except Exception as e:
            print(f"Error retrieving knowledge snippets: {e}")

        contexts.sort(key=lambda x: x['relevance_score'], reverse=True)
        return contexts[:n_results]

    def format_context(self, contexts: List[Dict]) -> str:
        if not contexts:
            return "No relevant context found."

        formatted_context = "Relevant Information:\n"

        for i, ctx in enumerate(contexts, 1):
            formatted_context += f"\n{i}. {ctx['content']}"

            if ctx['metadata'].get('product_id'):
                formatted_context += f" (Product ID: {ctx['metadata']['product_id']})"

            if ctx['metadata'].get('price'):
                formatted_context += f" (Price: ${ctx['metadata']['price']})"

            formatted_context += f" [Source: {ctx['source']}, Relevance: {ctx['relevance_score']:.2f}]"

        return formatted_context

    def format_user_profile(self, user_data: Dict) -> str:
        if not user_data:
            return "No user profile available."

        profile = f"User ID: {user_data.get('user_id', 'Unknown')}\n"

        if user_data.get('preferences'):
            prefs = user_data['preferences']
            profile += f"Preferred Categories: {', '.join(prefs.get('categories', []))}\n"
            profile += f"Preferred Brands: {', '.join(prefs.get('brands', []))}\n"
            profile += f"Price Range: ${prefs.get('min_price', 0)}-${prefs.get('max_price', 1000)}\n"

        if user_data.get('recent_interactions'):
            profile += f"Recent Activity: {len(user_data['recent_interactions'])} interactions\n"

        return profile

    def generate_rag_response(self, query: str, contexts: List[Dict], user_profile: str) -> str:
        if not self.llm:
            return self.generate_template_response(query, contexts, user_profile)

        context_text = self.format_context(contexts)

        prompt = self.rag_prompt.format(
            context=context_text,
            query=query,
            user_profile=user_profile
        )

        try:
            response = self.llm(prompt)
            return response
        except Exception as e:
            print(f"Error generating LLM response: {e}")
            return self.generate_template_response(query, contexts, user_profile)

    def generate_template_response(self, query: str, contexts: List[Dict], user_profile: str) -> str:
        if not contexts:
            return "I don't have enough information to provide specific recommendations. Please try a different query."

        product_contexts = [ctx for ctx in contexts if ctx['source'] == 'product_knowledge']

        if product_contexts:
            response = "Based on your query, I found these relevant products:\n\n"

            for ctx in product_contexts[:3]:
                metadata = ctx['metadata']
                response += f"• {metadata.get('product_id', 'Unknown Product')}"

                if metadata.get('category'):
                    response += f" - {metadata['category']}"

                if metadata.get('price'):
                    response += f" (${metadata['price']})"

                response += f"\n  {ctx['content'][:100]}...\n\n"

            response += "These recommendations are based on your preferences and similar user behaviors."
        else:
            response = f"I found some information related to '{query}':\n\n"
            response += self.format_context(contexts[:2])

        return response

    def execute(self, state: GraphState) -> GraphState:
        user_query = state.messages[-1] if state.messages else ""
        user_id = state.user_data.get('user_id') if state.user_data else None

        print(f"RAG Node processing query: {user_query}")

        contexts = self.retrieve_relevant_context(user_query, user_id=user_id, n_results=5)

        user_profile = self.format_user_profile(state.user_data or {})

        rag_response = self.generate_rag_response(user_query, contexts, user_profile)

        rag_metadata = {
            "contexts_found": len(contexts),
            "sources": list(set([ctx['source'] for ctx in contexts])),
            "relevance_scores": [ctx['relevance_score'] for ctx in contexts],
            "user_id": user_id
        }

        state.rag_context = contexts
        state.rag_response = rag_response
        state.rag_metadata = rag_metadata

        state.messages.append(f"RAG Context Retrieved: {len(contexts)} relevant items found")

        print(f"RAG Node completed. Found {len(contexts)} contexts, generated response length: {len(rag_response)}")

        return state

    def get_similar_products(self, product_id: str, n_results: int = 5) -> List[Dict]:
        try:
            results = self.kb_manager.get_similar_products(product_id, n_results=n_results)

            similar_products = []
            for i, doc in enumerate(results.get('documents', [[]])[0]):
                if i < len(results.get('metadatas', [[]])[0]):
                    metadata = results['metadatas'][0][i]
                    similar_products.append({
                        "content": doc,
                        "metadata": metadata,
                        "similarity_score": 1.0 - (i * 0.1)
                    })

            return similar_products
        except Exception as e:
            print(f"Error getting similar products: {e}")
            return []

    def update_user_context(self, user_id: str, interaction_data: Dict) -> None:
        try:
            interaction_text = f"User {user_id} recent interaction: "
            interaction_text += f"Product {interaction_data.get('product_id', 'Unknown')} - "
            interaction_text += f"Action: {interaction_data.get('action', 'unknown')} - "
            interaction_text += f"Rating: {interaction_data.get('rating', 'N/A')}"

            self.chroma_client.add_documents(
                "user_knowledge",
                [interaction_text],
                [{
                    "user_id": user_id,
                    "product_id": str(interaction_data.get('product_id', '')),
                    "action": interaction_data.get('action', 'unknown'),
                    "type": "recent_interaction"
                }],
                [f"recent_{user_id}_{interaction_data.get('product_id', 'unknown')}"]
            )

            print(f"Updated user context for {user_id}")
        except Exception as e:
            print(f"Error updating user context: {e}")

def create_rag_node(chroma_host: str = "localhost", chroma_port: int = 8001) -> RAGNode:
    try:
        chroma_client = ChromaDBClient(host=chroma_host, port=chroma_port)
        rag_node = RAGNode(chroma_client)

        print(f"✅ RAG Node created successfully, connected to ChromaDB at {chroma_host}:{chroma_port}")
        return rag_node

    except Exception as e:
        print(f"❌ Error creating RAG Node: {e}")
        raise
