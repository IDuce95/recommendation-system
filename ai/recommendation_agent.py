import logging
from typing import Dict, List, Optional
import time

import pandas as pd
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.constants import END, START
from langgraph.graph import StateGraph

from .graph_state import EnhancedGraphState
from .model_manager import ModelManager
from .mlflow_utils import MLflowRun, get_mlflow_tracker
from config.prompt_config import PromptConfig

try:
    from app.api.prometheus_metrics import monitor_recommendation
except ImportError:
    def monitor_recommendation(rec_type="default"):
        def decorator(func):
            return func
        return decorator

logger = logging.getLogger(__name__)


class RecommendationAgent:

    def __init__(self, model_manager: ModelManager, default_values: dict):
        self.model_manager = model_manager
        self.default_values = default_values
        self.prompt_config = PromptConfig()
        self.agent_chain = None
        self.agent_graph = None

    def initialize(self):
        logger.info("Initializing recommendation agent")
        self.agent_chain = self._create_agent_chain()

        self.agent_graph = self._create_enhanced_agent_graph()

        logger.info("Recommendation agent initialization complete")

    def format_recommended_products(self, recommended_products: pd.DataFrame) -> str:
        product_template = self.prompt_config.get_product_format_template()
        recommended_products_str = ""

        for _, product in recommended_products.iterrows():
            recommended_products_str += product_template.format(
                name=product.get('name', self.default_values["unknown_product"]),
                category=product.get('category', self.default_values["unknown_category"]),
                description=product.get('description', '').replace('\n', ', '),
                image_path=product.get('image_path', '')
            )
        return recommended_products_str

    def format_original_product_info(self, original_product: dict) -> dict:
        return {
            'original_product_name': original_product.get('name', self.default_values["unknown_product"]),
            'original_product_category': original_product.get('category', self.default_values["unknown_category"]),
            'original_product_description': original_product.get('description', '').replace('\n', ', ')
        }

    def _create_agent_chain(self):
        system_prompt = self.prompt_config.get_system_prompt()
        recommendation_template = self.prompt_config.get_recommendation_template()

        full_template = f"{system_prompt}\n\n{recommendation_template}"
        prompt_template = PromptTemplate.from_template(full_template)

        return (
            {
                "original_product_name": lambda x: x.get("original_product", {}).get('name', ''),
                "original_product_category": lambda x: x.get("original_product", {}).get('category', ''),
                "original_product_description": lambda x: x.get("original_product", {}).get('description', ''),
                "recommended_products": lambda x: self.format_recommended_products(x.get("recommended_products", pd.DataFrame()))
            }
            | prompt_template
            | self.model_manager.get_pipeline()
            | StrOutputParser()
        ).with_config({"run_name": "recommendations_agent"})

    def _create_enhanced_agent_graph(self):
        builder = StateGraph(state_schema=EnhancedGraphState)

        def text_generation_node(state: EnhancedGraphState) -> Dict:
            recommended_products = state.get('recommended_products', pd.DataFrame())
            original_product = state.get('original_product', {})

            input_dict = {
                "recommended_products": recommended_products,
                "original_product": original_product
            }

            response = self.agent_chain.invoke(input_dict)
            return {"recommendations_text": response}

        def image_formatting_node(state: EnhancedGraphState) -> Dict:
            recommendations_text = state.get('recommendations_text', '')
            recommended_products = state.get('recommended_products', pd.DataFrame())

            formatted_response = self._format_response_with_images(
                recommendations_text, recommended_products
            )

            return {"formatted_response": formatted_response}

        builder.add_node("generate_text", text_generation_node)
        builder.add_node("format_images", image_formatting_node)

        builder.add_edge(START, "generate_text")
        builder.add_edge("generate_text", "format_images")
        builder.add_edge("format_images", END)

        return builder.compile()

    def _format_response_with_images(self, text_response: str, recommended_products: pd.DataFrame) -> Dict:
        images = []
        for _, product in recommended_products.iterrows():
            image_info = {
                'product_name': product.get('name', ''),
                'image_path': product.get('image_path', ''),
                'category': product.get('category', '')
            }
            images.append(image_info)

        return {
            'text': text_response,
            'images': images,
            'product_count': len(recommended_products)
        }

    @monitor_recommendation("langgraph_workflow")
    def generate_recommendations(
        self,
        recommended_products: pd.DataFrame,
        chat_history: List[Dict],
        original_product: dict
    ) -> Dict:
        if self.agent_graph is None:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

        start_time = time.time()
        
        try:
            with MLflowRun(run_name="recommendation_generation") as tracker:
                tracker.log_params({
                    "original_product_name": original_product.get('name', 'unknown'),
                    "original_product_category": original_product.get('category', 'unknown'),
                    "num_recommended_products": len(recommended_products),
                    "chat_history_length": len(chat_history),
                    "model_name": self.model_manager.model_name
                })

                response = self.agent_graph.invoke({
                    "recommended_products": recommended_products,
                    "chat_history": chat_history,
                    "original_product": original_product
                })

                generation_time = time.time() - start_time
                result = response.get("formatted_response", {
                    "text": "No recommendations available",
                    "images": [],
                    "product_count": 0
                })

                tracker.log_metrics({
                    "generation_time_seconds": generation_time,
                    "output_product_count": result.get("product_count", 0),
                    "response_length_chars": len(result.get("text", "")),
                    "success": 1.0 if result.get("text") != "No recommendations available" else 0.0
                })

                logger.info(f"Generated recommendations in {generation_time:.2f}s with MLflow tracking")
                return result

        except Exception as e:
            logger.error(f"Error in recommendation generation: {e}")
            try:
                with MLflowRun(run_name="recommendation_generation_failed") as tracker:
                    tracker.log_params({
                        "original_product_name": original_product.get('name', 'unknown'),
                        "error": str(e)
                    })
                    tracker.log_metrics({
                        "generation_time_seconds": time.time() - start_time,
                        "success": 0.0
                    })
            except:
                pass  # Don't fail if MLflow logging fails
            
            return {
                "text": "Error generating recommendations",
                "images": [],
                "product_count": 0
            }
