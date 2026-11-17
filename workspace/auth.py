"""Prepare Streamlit Google OIDC configuration before server startup."""
import os
from urllib.parse import urlsplit
import toml
from workspace.settings import ROOT, load_environment


def prepare_auth():
    load_environment()
    path = ROOT / '.streamlit' / 'secrets.toml'
    config = toml.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    if not os.getenv('GOOGLE_CLIENT_ID') or not os.getenv('GOOGLE_CLIENT_SECRET'):
        return False
    cookie = os.getenv('AUTH_COOKIE_SECRET', '').strip()
    if len(cookie) < 32:
        raise ValueError('Set AUTH_COOKIE_SECRET to a stable random value of at least 32 characters.')
    redirect = os.getenv('AUTH_REDIRECT_URI', 'http://localhost:8501/oauth2callback')
    parsed = urlsplit(redirect)
    if parsed.path != '/oauth2callback' or (parsed.scheme != 'https' and
            not (parsed.scheme == 'http' and parsed.hostname in {'localhost', '127.0.0.1'})):
        raise ValueError('AUTH_REDIRECT_URI must be HTTPS with /oauth2callback, or HTTP localhost for development.')
    config['auth'] = {'redirect_uri': redirect, 'cookie_secret': cookie,
        'client_id': os.environ['GOOGLE_CLIENT_ID'], 'client_secret': os.environ['GOOGLE_CLIENT_SECRET'],
        'server_metadata_url': 'https://accounts.google.com/.well-known/openid-configuration'}
    path.parent.mkdir(exist_ok=True)
    path.write_text(toml.dumps(config), encoding='utf-8')
    return True
