import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline

logger = logging.getLogger(__name__)


class ModelManager:

    def __init__(self, model_config: dict):
        self.model_config = model_config
        self.model_name = model_config["name"]
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.hugging_face_pipeline = None

    def initialize_model(self):
        logger.info(f"Initializing model: {self.model_name}")

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.pipeline = pipeline(
            task=self.model_config["task"],
            model=self.model,
            tokenizer=self.tokenizer,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=self.model_config["max_new_tokens"],
            return_full_text=self.model_config["return_full_text"]
        )

        self.hugging_face_pipeline = HuggingFacePipeline(pipeline=self.pipeline)

        logger.info("Model initialization complete")

    def get_pipeline(self):
        if self.hugging_face_pipeline is None:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")
        return self.hugging_face_pipeline

    def get_raw_pipeline(self):
        if self.pipeline is None:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")
        return self.pipeline
