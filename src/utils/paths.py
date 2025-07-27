from pathlib import Path
import sys 



def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]

