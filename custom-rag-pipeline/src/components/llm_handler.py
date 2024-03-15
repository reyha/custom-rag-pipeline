"""
LLM handler for QA Pipeline
"""
from typing import Any

from llama_index.llms.llama_cpp import LlamaCPP

from src.utils.exceptions import CustomException, InvalidParametersError
from src.components.base_classes.base_module import BaseModule


class LLMHandler(BaseModule):
    SUPPORTED_MODELS = ['oss_llama-13b']

    def __init__(self, service_state: Any) -> None:
        super().__init__(service_state)

    def get_llm(self):
        model = self.inputs.get("model_id")
        if model not in self.SUPPORTED_MODELS:
            raise InvalidParametersError(f"Invalid model value: {model}")
        if model == "oss_llama-13b":
            llm = LlamaCPP(
                model_path=self.config['llm_model_path'],
                temperature=1.0,
                max_new_tokens=256,
                context_window=3900,
                generate_kwargs={},
                verbose=True,
            )
            return llm
        else:
            pass
        raise CustomException(
            500,
            "Issue in initialising llm"
        )
