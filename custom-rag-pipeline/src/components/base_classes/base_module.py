from typing import Any

class BaseModule:
    def __init__(self, service_state: Any) -> None:
        super().__init__()
        self.service_state = service_state
        # All the subsequent members should be updated by reference only.
        self.answer_id = service_state.answer_id
        self.app_start_time = service_state.app_start_time
        self.config = service_state.config
        self.logger = service_state.logger
        self.inputs = service_state.inputs
        self.interim_outputs = service_state.interim_outputs
