from pathlib import Path
from utils.paths import get_project_root
import pandas as pd

def NFLSQLDatabasePath():
     return get_project_root() / "data" / "nfl_data.db"
    