from typing import Tuple

import pandas as pd
import streamlit as st

from app.streamlit_app.config import (
    UI_LABELS, STATUS_MESSAGES, SLIDER_DEFAULTS, DEFAULT_WEIGHTS,
    DEFAULT_VALUES, IMAGE_CONFIG
)


class WeightConfigurationUI:

    @staticmethod
    def create_weight_slider() -> Tuple[float, float, int, int]:
        st.markdown(UI_LABELS["weights_header"])
        st.write(STATUS_MESSAGES["weight_description"])

        text_weight_percent = st.slider(
            UI_LABELS["text_weight_slider"],
            SLIDER_DEFAULTS["text_weight_percent"]["min_value"],
            SLIDER_DEFAULTS["text_weight_percent"]["max_value"],
            SLIDER_DEFAULTS["text_weight_percent"]["default_value"],
            SLIDER_DEFAULTS["text_weight_percent"]["step"],
            help=SLIDER_DEFAULTS["text_weight_percent"]["help_text"]
        )

        image_weight_percent = 100 - text_weight_percent
        text_weight = text_weight_percent / 100.0
        image_weight = image_weight_percent / 100.0

        return text_weight, image_weight, text_weight_percent, image_weight_percent

    @staticmethod
    def display_weight_metrics(text_weight_percent: int, image_weight_percent: int) -> None:
        st.write(STATUS_MESSAGES["weight_distribution"])
        col_text, col_image = st.columns(2)
        with col_text:
            st.metric(label=UI_LABELS["text_weight_metric"], value=f"{text_weight_percent}%")
        with col_image:
            st.metric(label=UI_LABELS["image_weight_metric"], value=f"{image_weight_percent}%")

    @classmethod
    def handle_dual_embeddings(cls) -> Tuple[float, float]:
        text_weight, image_weight, text_weight_percent, image_weight_percent = cls.create_weight_slider()
        cls.display_weight_metrics(text_weight_percent, image_weight_percent)
        return text_weight, image_weight

    @staticmethod
    def handle_single_embedding_type(use_text_embeddings: bool) -> Tuple[float, float]:
        if use_text_embeddings:
            st.info(STATUS_MESSAGES["info_text_only"])
            return DEFAULT_WEIGHTS["text_only"]["text"], DEFAULT_WEIGHTS["text_only"]["image"]
        else:
            st.info(STATUS_MESSAGES["info_image_only"])
            return DEFAULT_WEIGHTS["image_only"]["text"], DEFAULT_WEIGHTS["image_only"]["image"]

    @classmethod
    def handle_weight_configuration(cls, use_text_embeddings: bool, use_image_embeddings: bool) -> Tuple[float, float]:
        if use_text_embeddings and use_image_embeddings:
            return cls.handle_dual_embeddings()
        else:
            return cls.handle_single_embedding_type(use_text_embeddings)


class ProductSelectionUI:

    @staticmethod
    def create_category_selector(products: pd.DataFrame) -> str:
        return st.selectbox(
            label=UI_LABELS["category_select"],
            options=products['category'].unique().tolist(),
            index=None
        )

    @staticmethod
    def create_product_selector(filtered_products: pd.DataFrame) -> str:
        product_options = [
            f"{i + 1}. {filtered_products.iloc[i]['name']} ({filtered_products.iloc[i]['description'].replace('\n', ', ')})"
            for i in range(len(filtered_products))
        ]
        return st.selectbox(
            label=UI_LABELS["product_select"],
            options=product_options,
            index=None
        )

    @staticmethod
    def display_selected_product(selected_product_details: pd.Series) -> None:
        st.markdown(UI_LABELS["selected_product_header"])
        st.markdown(f'##### {selected_product_details.get("name", "Unknown Product")}')
        st.write(selected_product_details.get("description", "No description available").replace('\n', '  \n'))

        if 'image_path' in selected_product_details and pd.notna(selected_product_details.get('image_path', '')):
            st.image(selected_product_details.get('image_path', ''), width=IMAGE_CONFIG["width"])


class EmbeddingSelectionUI:

    @staticmethod
    def create_embedding_checkboxes() -> Tuple[bool, bool]:
        st.markdown(UI_LABELS["embeddings_header"])
        use_text_embeddings = st.checkbox(
            UI_LABELS["text_embeddings_checkbox"],
            value=DEFAULT_VALUES["use_text_embeddings"]
        )
        use_image_embeddings = st.checkbox(
            UI_LABELS["image_embeddings_checkbox"],
            value=DEFAULT_VALUES["use_image_embeddings"]
        )
        return use_text_embeddings, use_image_embeddings

    @staticmethod
    def validate_embeddings_selection(use_text_embeddings: bool, use_image_embeddings: bool) -> Tuple[bool, float, float]:
        if not use_text_embeddings and not use_image_embeddings:
            st.warning(STATUS_MESSAGES["warning_no_embeddings"])
            use_text_embeddings = True
            text_weight = DEFAULT_WEIGHTS["text_only"]["text"]
            image_weight = DEFAULT_WEIGHTS["text_only"]["image"]
            return use_text_embeddings, text_weight, image_weight

        text_weight, image_weight = WeightConfigurationUI.handle_weight_configuration(
            use_text_embeddings, use_image_embeddings
        )
        return use_text_embeddings, text_weight, image_weight


class RecommendationUI:

    @staticmethod
    def create_recommendations_slider() -> int:
        st.markdown(UI_LABELS["recommendations_count_header"])
        return st.slider(
            UI_LABELS["recommendations_count_slider"],
            min_value=SLIDER_DEFAULTS["top_n_recommendations"]["min_value"],
            max_value=SLIDER_DEFAULTS["top_n_recommendations"]["max_value"],
            value=SLIDER_DEFAULTS["top_n_recommendations"]["default_value"],
            step=SLIDER_DEFAULTS["top_n_recommendations"]["step"],
            help=SLIDER_DEFAULTS["top_n_recommendations"]["help_text"]
        )
