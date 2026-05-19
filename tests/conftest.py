import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

sqlite_path = (ROOT_DIR / 'test.db').resolve().as_posix()
os.environ.setdefault("DATABASE_URL", f"sqlite:///{sqlite_path}")
