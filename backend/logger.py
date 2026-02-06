import sys
import io
from backend.global_state import state

import re

class StreamToLogger(io.TextIOBase):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, original_stream):
        self.original_stream = original_stream

    def write(self, buf):
        if buf.strip():
            # Strip ANSI color codes
            clean_buf = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', buf)
            state.add_log(clean_buf.strip())
        self.original_stream.write(buf)
        return len(buf)
        
    def flush(self):
        self.original_stream.flush()

def setup_logger():
    # Redirect stdout and stderr
    sys.stdout = StreamToLogger(sys.stdout)
    sys.stderr = StreamToLogger(sys.stderr)
