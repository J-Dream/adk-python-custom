import sys
import os

# Add the project root to the Python path to allow imports from app and config
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.main import main

if __name__ == "__main__":
    print("Executing run.py")
    main()
