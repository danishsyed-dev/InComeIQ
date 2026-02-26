"""
Custom exception handler with detailed error information.

Wraps standard Python exceptions with the originating filename
and line number for easier debugging.
"""

import sys


def error_message_detail(error: Exception, error_detail: sys) -> str:
    """Build a detailed error message including file and line number."""
    _, _, exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    error_message = (
        f"Error in script [{file_name}] "
        f"at line [{line_number}]: {str(error)}"
    )
    return error_message


class CustomException(Exception):
    """Exception that includes file name and line number in its message."""

    def __init__(self, error_message: Exception, error_detail: sys):
        super().__init__(str(error_message))
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self) -> str:
        return self.error_message
