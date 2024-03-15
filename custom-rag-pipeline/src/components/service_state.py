"""
Service state for QA pipeline
"""

from typing import Any


class ServiceState:
    def __init__(
        self,
        answer_id: str,
        app_start_time: Any,
        config: dict,
        inputs: dict,
        vector_store: Any,
        logger: Any,
        interim_outputs: dict=None
    ) -> None:
        self.answer_id = answer_id
        self.app_start_time = app_start_time
        self.inputs = inputs
        self.config = config
        self.logger = logger
        self.vector_store = vector_store
        if interim_outputs is not None:
            self.interim_outputs = interim_outputs
        else:
            self.interim_outputs = {}
