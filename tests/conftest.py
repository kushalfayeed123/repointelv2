
import os
import sys

# Track the project root directory (one level up from tests/)
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Inject it into the front of Python's module resolution array
if root_path not in sys.path:
    sys.path.insert(0, root_path)