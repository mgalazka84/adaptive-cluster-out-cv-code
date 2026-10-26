"""Read benchmark CSVs, including the gzip archives published on GitHub."""
from pathlib import Path
import pandas as pd

def read_csv(path, **kwargs):
    path = Path(path)
    if not path.exists():
        path = path.with_suffix(path.suffix + ".gz")
    return pd.read_csv(path, **kwargs)
