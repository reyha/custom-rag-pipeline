import time
import json
import uuid
import traceback
from flask import Flask, Response, request

from src.utils.logger import get_logger
from src.utils.configuration import Configuration
from src.utils.exceptions import CustomException, ValidationException
from src.components.service_state import ServiceState
from src.answer_generator import QAGenerator
from src.components.data_handler import DataHandler


# =================== Initializations ===================
# 0. Initialize Flask
app = Flask(__name__)

# 1. Initialize configs
base_configuration = Configuration("config/settings.toml")
dev_configuration = Configuration("config/dev_settings.toml").get_config(not_env_specific=True)

# 1.1 Get default config and environment specific config
app_configuration = base_configuration.get_config()

# 2. Initialize logger
logger = get_logger("customragpipeline" "INFO")

# 3. Create vector store
data_handler = DataHandler(app_configuration)
vector_store = data_handler.create_vector_store()


def validate_request(request_data):
    # A. Mandatory fields
    # 1. Validate mandatory fields
    mandatory_fields = ['user_query']
    for field in mandatory_fields:
        if field not in request_data.keys():
            raise ValidationException(400, f"{field} field missing")

    # Note: Any of the fields can be sent as json null - So, check for not field
    validated_fields = {}

    # 2. User query check
    user_query = request_data.get('user_query')
    if (not user_query) or (not isinstance(user_query, str)) or (not user_query.strip()):
        raise ValidationException(422, "user_query cannot be empty string")
    validated_fields["user_query"] = user_query

    # 3. Model ID
    model_id = request_data.get('model_id', None)
    if (not model_id) or (not isinstance(model_id, str)):
        logger.warning(f"answer_id:{model_id}:Defaulting Model")
        model_id = "oss_llama-13b"
    validated_fields["model_id"] = model_id

    return validated_fields


@app.route('/v1/custom_rag_qna', methods=['POST'])
def get_qna_response():
    """Define the endpoint for QA Response
    Returns:
        flask.Response: A response object containing
                        helpful and relevant responses.
    """
    # Generate answer_id
    answer_id = str(uuid.uuid1())
    logger.info(f"answer_id:{answer_id}")

    try:
        logger.info(f"Starting to serve request:answer_id:{answer_id}")
        app_start_time = time.time()

        # =================== Read Data and Validate ===================
        # 1. Request Body
        request_data = json.loads(request.data)

        # 1.1 Validate fields
        validated_fields = validate_request(request_data)

        # ==============================================================
        service_state = ServiceState(answer_id=answer_id,
                                     app_start_time=app_start_time,
                                     config={**dev_configuration, **app_configuration},
                                     inputs={**validated_fields},
                                     vector_store=vector_store,
                                     logger=logger)
        # 2. Initialize QA Generator
        qa = QAGenerator(service_state)

        # 3. Prepare context/relevant docs and LLM
        qa.prepare()

        # 4. Generate response
        return Response(qa.generate())

    except ValidationException as validation_exception:
        logger.error(f"answer_id:{answer_id}:"
                     f"ValidationError:"
                     f"{validation_exception.response_code} - {validation_exception.response_msg}")
        error_payload = json.dumps({
            "name": "VALIDATION_ERROR",
            "message": validation_exception.response_msg,
            "debug_id": answer_id
        })
        return Response(response=error_payload,
                        status=validation_exception.response_code,
                        mimetype="application/json"
                        )

    except CustomException as custom_exception:
        traceback_str = traceback.format_exc().replace('\n', ';')
        logger.error(f"answer_id:{answer_id}:"
                     f"CustomException: "
                     f"{custom_exception.response_code} - {custom_exception.response_msg} :"
                     f"{traceback_str}")
        error_payload = json.dumps({
            "name": "CUSTOM_EXCEPTION",
            "message": custom_exception.response_msg,
            "debug_id": answer_id
        })

        return Response(response=error_payload, status=custom_exception.response_code,
                        mimetype="application/json")
    except Exception as exception:
        traceback_str = traceback.format_exc().replace('\n', ';')
        logger.error(f"InternalError: {exception} :"
                     f"{traceback_str}")
        error_payload = json.dumps({
            "name": "INTERNAL_EXCEPTION",
            "message": str(exception),
            "debug_id": answer_id
        })
        return Response(response=error_payload, status=500, mimetype="application/json")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=app_configuration.get("PORT"), debug=False)
