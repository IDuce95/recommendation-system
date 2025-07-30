from typing import Dict, List, Optional, Any
from typing_extensions import TypedDict
import pandas as pd

class GraphState(TypedDict):
    recommended_products: pd.DataFrame
    chat_history: List[Dict[str, str]]
    recommendations_text: Optional[str]

class EnhancedGraphState(TypedDict):
    recommended_products: pd.DataFrame
    chat_history: List[Dict[str, str]]
    original_product: Dict[str, Any]
    recommendations_text: Optional[str]
    formatted_response: Optional[Dict[str, Any]]
    messages: List[str]
    user_data: Optional[Dict[str, Any]]
    rag_context: Optional[List[Dict]]
    rag_response: Optional[str]
    rag_metadata: Optional[Dict[str, Any]]
