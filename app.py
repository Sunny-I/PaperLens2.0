"""PaperLens: private projects, persistent PDFs and cited research chat."""
import json
import os
import streamlit as st
from sqlalchemy.exc import SQLAlchemyError
from workspace.settings import load_environment, missing_configuration, integer
from workspace.database import Repository
from workspace.service import WorkspaceService
from workspace.ui import apply_workspace_style, display_name

load_environment()
st.set_page_config(page_title='PaperLens', page_icon='📄', layout='wide')
apply_workspace_style()


@st.cache_resource
def repository():
    repo = Repository()
    repo.initialize()
    return repo


def report_error(exc):
    if isinstance(exc, ValueError):
        st.error(str(exc))
    elif isinstance(exc, SQLAlchemyError):
        st.error('Database operation failed. Check the PostgreSQL connection and try again.')
    else:
        st.error('The operation could not finish. Check server configuration and try again.')


def render_answer(payload, service, owner, project_id, key):
    result = payload['result']
    st.caption('AI answer · sources checked' if result['mode'] == 'generated' else 'Source evidence')
    st.write(result['text'])
    if result.get('warning'):
        warning = result['warning']
        if 'validation' in warning.lower():
            st.warning('The AI answer could not pass source checks. Read the matching passages below, or ask again.')
            with st.expander('Why this answer fell back to sources'):
                st.caption(warning.replace(' Showing source excerpts instead.', '').strip())
        else:
            st.warning(warning)
    if result['mode'] == 'extractive':
        st.caption(f"{len(payload['hits'])} matching passages. Expand a source below to read it.")
    else:
        for claim in result['claims']:
            st.write(claim['text'])
            labels = []
            for source_id in sorted({c['source_id'] for c in claim['citations']}):
                if 1 <= source_id <= len(payload['hits']):
                    source = payload['hits'][source_id - 1]
                    labels.append(f"[{source_id}] {source['filename']} · page {source['page']}")
            st.caption('Sources: ' + '  |  '.join(labels))
    for index, hit in enumerate(payload['hits'], 1):
        with st.expander(f"[{index}] {hit['filename']} · page {hit['page']}"):
            quotes = list(dict.fromkeys(c['quote'] for claim in result['claims']
                for c in claim['citations'] if c['source_id'] == index))
            st.text('\n\n'.join(quotes) if quotes else hit['text'])
            if st.button('Open original PDF', key=f'source-{key}-{index}'):
                try:
                    st.link_button('View PDF at this page', service.source_url(owner, project_id,
                        hit['doc_id'], hit['page']))
                except ValueError as exc:
                    st.info(str(exc))
    if result['mode'] == 'generated':
        st.caption('Citations verify where quotes came from. Review the source to confirm the interpretation.')


missing = missing_configuration()
if missing:
    st.title('📄 PaperLens')
    st.subheader('Your research workspace is almost ready')
    st.write('Fill in the local .env file, then restart with the command below.')
    st.code('.venv\\Scripts\\python scripts/run.py', language='powershell')
    st.write('Missing settings:')
    st.code('\n'.join(missing), language=None)
    st.caption('Credentials stay on the server. Do not paste them into chat.')
    st.stop()

if not st.user.is_logged_in:
    st.title('📄 PaperLens')
    st.header('Your papers. Clearer answers.')
    st.write('Create a research project, add your PDFs, and explore answers backed by the original sources.')
    st.button('Continue with Google', type='primary', on_click=st.login)
    st.caption('Your projects, papers, and conversations are private to your account.')
    st.stop()

subject = st.user.get('sub')
if not subject:
    st.error('Google did not return a valid account identity. Sign out and try again.')
    st.button('Sign out', on_click=st.logout)
    st.stop()

try:
    repo = repository()
    user = repo.user(subject, st.user.get('email', ''), st.user.get('name', 'Researcher'))
    service = WorkspaceService(repo)
    project_list = repo.list_projects(user['id'])
except Exception as exc:
    report_error(exc)
    st.stop()

owner = user['id']
with st.sidebar:
    st.title('📄 PaperLens')
    st.caption(f"Signed in as {display_name(user)}")
    st.button('Sign out', on_click=st.logout, use_container_width=True)
    st.divider()
    st.subheader('My projects')
    with st.form('new-project', clear_on_submit=True):
        name = st.text_input('Project name', max_chars=120, placeholder='e.g. Medical image analysis')
        if st.form_submit_button('＋ Create project', use_container_width=True):
            try:
                project_id = repo.create_project(owner, name,
                    os.getenv('OPENAI_EMBED_MODEL', 'text-embedding-3-small'), integer('OPENAI_EMBED_DIMENSIONS', 1536))
                st.session_state['active_project'] = project_id
                st.rerun()
            except Exception as exc:
                report_error(exc)
    ids = [p['id'] for p in project_list]
    labels = {p['id']: p['name'] for p in project_list}
    if ids:
        if st.session_state.get('active_project') not in ids:
            st.session_state['active_project'] = ids[0]
        project_id = st.selectbox('Open project', ids, format_func=lambda key: labels[key], key='active_project')
    st.divider()
    st.caption('Your papers and conversations are saved automatically. Each project has its own research library.')

if not project_list:
    st.title('A home for every research question')
    st.write('Create your first project in the sidebar. Each project keeps its own papers, citations, and conversation.')
    st.stop()

try:
    project = repo.project(owner, project_id)
    records = repo.list_files(owner, project_id)
except Exception as exc:
    report_error(exc)
    st.stop()
ready = [record for record in records if record['status'] == 'ready']
st.title(project['name'])
st.caption('A private collection of papers, evidence, and ideas.')
c1, c2, c3 = st.columns(3)
c1.metric('Papers ready', len(ready))
c2.metric('Pages', sum(record['total_pages'] for record in ready))
c3.metric('Searchable passages', sum(record['chunk_count'] for record in ready))
files_tab, chat_tab, summary_tab = st.tabs(['Files', 'Chat', 'Summaries'])

with files_tab:
    st.subheader('Build your research library')
    st.caption(f"Upload text-based PDFs up to {integer('MAX_FILE_MB', 20)} MB each. "
               'Files are embedded once and stay available after refresh. Scanned PDFs need OCR first.')
    upload_key = f"uploads-{project_id}-{st.session_state.get('upload_version', 0)}"
    uploads = st.file_uploader('Add papers', type=['pdf'], accept_multiple_files=True, key=upload_key)
    st.caption('Paper passages are sent to the configured embedding service and stored for search.')
    if st.button('Upload and analyze', type='primary', disabled=not uploads):
        progress = st.progress(0, text='Preparing PDFs…')
        for index, upload in enumerate(uploads):
            progress.progress(index / len(uploads), text=f'Processing {upload.name}…')
            try:
                service.upload(owner, project_id, upload.name, upload.getvalue())
                st.success(f'{upload.name} is ready.')
            except Exception as exc:
                report_error(exc)
        progress.progress(1.0, text='Processing finished')
        st.session_state['upload_version'] = st.session_state.get('upload_version', 0) + 1
        records = repo.list_files(owner, project_id)
    if not records:
        st.info('Add your first PDF to start exploring this project.')
    for record in records:
        with st.container(border=True):
            left, right = st.columns([4, 1])
            left.write(record['filename'])
            left.caption(f"{record['status'].capitalize()} · {record['total_pages']} pages · "
                         f"{record['size_bytes'] / 1024:.0f} KB")
            if record['error']:
                left.warning(record['error'])
            if record['status'] == 'ready' and left.button('Open PDF', key='open-' + record['id']):
                try:
                    left.link_button('View original', service.source_url(owner, project_id, record['id']))
                except Exception as exc:
                    report_error(exc)
            confirm = right.checkbox('Confirm removal', key='confirm-' + record['id'])
            if right.button('Remove', key='remove-' + record['id'], disabled=not confirm):
                try:
                    service.remove_file(owner, project_id, record['id'])
                    st.rerun()
                except Exception as exc:
                    report_error(exc)

with chat_tab:
    st.subheader('Ask your papers')
    st.caption('Answers use evidence from this project. Earlier messages help interpret follow-up questions.')
    ready = [r for r in records if r['status'] == 'ready']
    selected = st.multiselect('Search specific papers (empty means all)',
        [r['id'] for r in ready], format_func=lambda key: next(r['filename'] for r in ready if r['id'] == key),
        key=f'scope-{project_id}')
    generate = st.toggle('Write AI answers', value=True, key=f'generate-{project_id}',
                         help='When off, embedding API calls still find original passages.')
    try:
        history = repo.history(owner, project_id, limit=100)
        if not history:
            st.info('Try “What problem do these papers address?” or “Compare their methods and limitations.”')
        for message in history:
            with st.chat_message(message['role']):
                if message['role'] == 'user':
                    st.write(message['content'])
                else:
                    render_answer(message['payload'], service, owner, project_id, message['id'])
        if history:
            st.download_button('Export displayed conversation', json.dumps(history, default=str, indent=2),
                file_name=f'paperlens-{project_id}-chat.json', mime='application/json')
        question = st.chat_input('Ask a question about this project…', disabled=not ready, max_chars=2000)
        if question:
            with st.spinner('Finding evidence and checking citations…'):
                service.ask(owner, project_id, question, selected or None, generate)
            st.rerun()
    except Exception as exc:
        report_error(exc)

with summary_tab:
    st.subheader('From long papers to clear findings')
    if not ready:
        st.info('Upload a paper before generating a summary.')
    for record in ready:
        with st.container(border=True):
            st.write(record['filename'])
            if st.button('Generate summary' if not record['summary'] else 'Refresh summary',
                         key='summary-' + record['id']):
                try:
                    with st.spinner('Reading all pages and validating supporting quotes…'):
                        service.summarize(owner, project_id, record['id'])
                    st.rerun()
                except Exception as exc:
                    report_error(exc)
            if record['summary']:
                saved = record['summary']
                st.caption('AI summary' if saved['mode'] == 'generated' else 'Source excerpts')
                if saved.get('warning'):
                    st.warning(saved['warning'])
                summary = saved['summary']
                for field, label in [('research_problem', 'Research problem'), ('methodology', 'Method'),
                                     ('datasets_and_benchmarks', 'Datasets and experiments'),
                                     ('key_findings', 'Findings'), ('limitations', 'Limitations')]:
                    st.markdown(f'**{label}**')
                    value = summary.get(field)
                    st.write('\n\n'.join(value) if isinstance(value, list) and value else value or 'Not found in the paper.')
                    evidence = summary.get('evidence', {}).get(field, [])
                    if evidence:
                        with st.expander(f'Sources for {label.lower()}'):
                            for quote in evidence:
                                st.text(quote['quote'])
                                st.caption(f"{record['filename']} · page {quote['page']}")
                st.download_button('Download summary', json.dumps(saved, indent=2),
                    file_name=f"summary-{record['id']}.json", mime='application/json', key='export-' + record['id'])
