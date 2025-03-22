import os
import sys
from pathlib import Path

# Get the absolute path to the project root
project_root = Path(__file__).parent.parent.absolute()

# Add project root and src to Python path
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Print debug information
print(f"Project root: {project_root}")
print(f"Python path: {sys.path}")
print(f"Contents of src: {os.listdir(str(project_root / 'src'))}")