"""Theme-neutral presentation helpers for the research workspace.

Colors belong to Streamlit's light/dark palettes, not CSS overrides. This keeps
widgets, menus and their disabled/focus states in the same active theme.
"""
import streamlit as st


WORKSPACE_CSS = """
<style>
.block-container {
    max-width: 1180px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}
h1, h2, h3 { letter-spacing: -.035em; }
h1 { line-height: 1.15; }
[data-testid="stSidebar"] {
    border-right: 1px solid color-mix(in srgb, currentColor 12%, transparent);
}
[data-testid="stSidebarUserContent"] { padding-top: 1.4rem; }
[data-testid="stSidebar"] [data-testid="stHeadingWithActionElements"] h1 {
    font-size: 1.65rem;
}
[data-testid="stSidebar"] hr { margin: 1.35rem 0; }
[data-testid="stSidebar"] [data-testid="stForm"] {
    padding: 1rem;
    border-radius: 14px;
}
[data-testid="stMetric"] {
    border: 1px solid color-mix(in srgb, currentColor 16%, transparent);
    border-radius: 14px;
    padding: 1rem 1.15rem;
}
[data-testid="stMetricValue"] { font-weight: 650; }
[data-testid="stChatMessage"] { border-radius: 14px; padding: 1.1rem 1.25rem; }
[data-testid="stChatMessageContent"] { min-width: 0; }
[data-testid="stMarkdownContainer"] { overflow-wrap: anywhere; }
[data-testid="stText"] { white-space: pre-wrap; overflow-wrap: anywhere; }
[data-testid="stExpander"] { border-radius: 12px; }
[data-testid="stButton"] button,
[data-testid="stFormSubmitButton"] button,
[data-testid="stLinkButton"] a { min-height: 2.65rem; border-radius: 10px; }
[data-testid="stButton"] button:focus-visible,
[data-testid="stFormSubmitButton"] button:focus-visible,
[data-testid="stLinkButton"] a:focus-visible {
    outline: 2px solid currentColor;
    outline-offset: 3px;
}
@media (max-width: 640px) {
    .block-container { padding-top: 1.2rem; padding-left: 1rem; padding-right: 1rem; }
    [data-testid="stMetric"] { padding: .8rem; }
    [data-testid="stChatMessage"] { padding: .85rem; }
}
</style>
"""


def apply_workspace_style():
    st.markdown(WORKSPACE_CSS, unsafe_allow_html=True)


def display_name(user):
    """Avoid displaying placeholder identity names returned by Google profiles."""
    name = (user.get('name') or '').strip()
    if name.casefold() not in {'', 'unknown', 'none', 'null'}:
        return name
    email = (user.get('email') or '').strip()
    return email.split('@', 1)[0] if email else 'Researcher'
