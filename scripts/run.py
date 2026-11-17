"""Load .env, prepare Google OIDC, then start the server without logging secrets."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workspace.auth import prepare_auth
from workspace.settings import load_environment, integer

if __name__ == '__main__':
    load_environment()
    prepare_auth()
    port = integer('PORT', 8501)
    raise SystemExit(subprocess.call([sys.executable, '-m', 'streamlit', 'run', str(ROOT / 'app.py'),
        '--server.port', str(port), '--server.address', os.getenv('BIND_ADDRESS', '127.0.0.1'),
        '--server.headless', 'true'], cwd=ROOT))
