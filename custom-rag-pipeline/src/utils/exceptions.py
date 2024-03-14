class InvalidParametersError(Exception):
    """Define error type for invalid parameters."""


class CustomException(Exception):
    # Constructor or Initializer
    def __init__(self, code, message):
        self.response_code = code
        self.response_msg = message

    # __str__ is to print() the value
    def __str__(self):
        return f"{self.__class__.__name__}: {self.response_code} - {self.response_msg}"


class ValidationException(CustomException):
    """ To be used exclusively for Validation Exceptions"""
