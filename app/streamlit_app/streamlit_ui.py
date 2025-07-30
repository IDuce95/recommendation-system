import os
import sys
from typing import Dict

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import pandas as pd
import streamlit as st

from app.data_processing.data_loader import DataLoader
from config.config_manager import get_config
from app.streamlit_app.config import STATUS_MESSAGES
from app.streamlit_app.api_client import RecommendationAPIClient
from app.streamlit_app.ui_components import (
    ProductSelectionUI, EmbeddingSelectionUI, RecommendationUI
)

PAGE_TITLE = "Product recommendation system"
PAGE_LAYOUT = "wide"
MAIN_COLUMNS_SPEC = [0.4, 0.6]

st.set_page_config(page_title=PAGE_TITLE, layout=PAGE_LAYOUT)
st.title(PAGE_TITLE)

config = get_config()
api_config = config.get_api_config()
api_client = RecommendationAPIClient(
    base_url=f"http://{api_config['host']}:{api_config['port']}",
    endpoint="/api/get_recommendations/"
)

if "recommendations_data" not in st.session_state:
    st.session_state.recommendations_data = None

col1, col2 = st.columns(spec=MAIN_COLUMNS_SPEC)

@st.cache_resource
def load_product_data() -> pd.DataFrame:
    try:
        data_loader = DataLoader()
        product_data = data_loader.load_product_data()
        if product_data is None:
            return pd.DataFrame()
        return product_data
    except Exception as e:
        st.error(STATUS_MESSAGES["error_db"].format(error=e))
        return pd.DataFrame()

def extract_product_info(selected_product: str, filtered_products: pd.DataFrame) -> Dict:
    selected_product_index = int(selected_product.split('.')[0]) - 1
    return filtered_products.iloc[selected_product_index].to_dict()

def process_api_response(recommendations_data: Dict) -> None:
    if recommendations_data and recommendations_data.get("success", False):
        st.session_state.recommendations_data = recommendations_data
        st.success(STATUS_MESSAGES["success"])
    else:
        st.error(STATUS_MESSAGES["error_api"])
        st.session_state.recommendations_data = None

def display_recommendations() -> None:
    if st.session_state.recommendations_data is None:
        st.info("Select a product and generate recommendations to see results here.")
        return

    recommendations_data = st.session_state.recommendations_data

    if "recommendations_text" in recommendations_data:
        st.markdown("### Product recommendations")

        recommendations_text = recommendations_data["recommendations_text"]
        images = recommendations_data.get("images", [])

        if "🔸" in recommendations_text:
            parts = recommendations_text.split("🔸")

            if parts[0].strip():
                st.markdown(parts[0].strip())

            for i, part in enumerate(parts[1:], 0):
                if part.strip():
                    col_text, col_img = st.columns([2, 1])

                    with col_text:
                        st.markdown(f"🔸{part.strip()}")

                    with col_img:
                        if i < len(images):
                            image_info = images[i]
                            image_path = image_info.get('image_path') if isinstance(image_info, dict) else getattr(image_info, 'image_path', '')
                            product_name = image_info.get('product_name', 'Product') if isinstance(image_info, dict) else getattr(image_info, 'product_name', 'Product')

                            if image_path:
                                try:
                                    st.image(
                                        image_path,
                                        caption=product_name,
                                        width=200
                                    )
                                except Exception as e:
                                    st.error(f"Could not load image: {e}")
        else:
            st.write(recommendations_text)

            if images:
                st.markdown("### Product images")
                cols_per_row = 3

                for i in range(0, len(images), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, col in enumerate(cols):
                        if i + j < len(images):
                            image_info = images[i + j]
                            with col:
                                image_path = image_info.get('image_path') if isinstance(image_info, dict) else getattr(image_info, 'image_path', '')
                                product_name = image_info.get('product_name', 'Product') if isinstance(image_info, dict) else getattr(image_info, 'product_name', 'Product')
                                category = image_info.get('category', 'Unknown') if isinstance(image_info, dict) else getattr(image_info, 'category', 'Unknown')

                                if image_path:
                                    try:
                                        st.image(
                                            image_path,
                                            caption=f"{product_name} ({category})",
                                            use_container_width=True
                                        )
                                    except Exception as e:
                                        st.error(f"Could not load image: {e}")

def handle_recommendation_generation(
    selected_product: str,
    filtered_products: pd.DataFrame,
    use_text_embeddings: bool,
    use_image_embeddings: bool,
    top_n: int,
    text_weight: float,
    image_weight: float,
) -> None:
    if not selected_product:
        st.warning(STATUS_MESSAGES["warning_no_product"])
        return

    selected_product_info = extract_product_info(selected_product, filtered_products)

    with st.spinner(STATUS_MESSAGES["loading_recommendations"]):
        recommendations_data = api_client.get_recommendations(
            selected_product_info=selected_product_info,
            use_text_embeddings=use_text_embeddings,
            use_image_embeddings=use_image_embeddings,
            top_n=top_n,
            text_weight=text_weight,
            image_weight=image_weight
        )
        process_api_response(recommendations_data)

def main():
    products = load_product_data()

    if products.empty:
        st.error("No product data available.")
        return

    with col1.container(border=True):
        st.markdown("### Settings")

        category = ProductSelectionUI.create_category_selector(products)
        selected_product = None
        filtered_products = pd.DataFrame()

        if category:
            filtered_products = products[products['category'] == category]
            selected_product = ProductSelectionUI.create_product_selector(filtered_products)

            if selected_product:
                selected_product_index = int(selected_product.split('.')[0]) - 1
                selected_product_details = filtered_products.iloc[selected_product_index]
                ProductSelectionUI.display_selected_product(selected_product_details)

        use_text_embeddings, use_image_embeddings = EmbeddingSelectionUI.create_embedding_checkboxes()

        top_n = RecommendationUI.create_recommendations_slider()

        use_text_embeddings, text_weight, image_weight = EmbeddingSelectionUI.validate_embeddings_selection(
            use_text_embeddings, use_image_embeddings
        )

        button_disabled = not (category and selected_product)

        if st.button("Generate Recommendations", disabled=button_disabled):
            handle_recommendation_generation(
                selected_product,
                filtered_products,
                use_text_embeddings,
                use_image_embeddings,
                top_n,
                text_weight,
                image_weight
            )

    with col2.container(border=True):
        display_recommendations()

if __name__ == "__main__":
    main()
