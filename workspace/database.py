"""Relational records. Every project operation checks the authenticated owner."""
import uuid
from datetime import datetime, timezone

from sqlalchemy import (MetaData, Table, Column, String, Text, Integer, DateTime,
                        ForeignKey, UniqueConstraint, JSON, create_engine, select, func)
from sqlalchemy.dialects.postgresql import JSONB

from workspace.settings import database_url, integer

metadata = MetaData()
json_type = JSON().with_variant(JSONB(), 'postgresql')


def uid():
    return uuid.uuid4().hex


def now():
    return datetime.now(timezone.utc)


users = Table('pl_users', metadata,
    Column('id', String(32), primary_key=True),
    Column('identity', String(300), nullable=False, unique=True),
    Column('email', String(320), nullable=False),
    Column('name', String(200), nullable=False),
    Column('created_at', DateTime(timezone=True), nullable=False))

projects = Table('pl_projects', metadata,
    Column('id', String(32), primary_key=True),
    Column('user_id', ForeignKey('pl_users.id'), nullable=False, index=True),
    Column('name', String(120), nullable=False),
    Column('namespace', String(80), nullable=False, unique=True),
    Column('embedding_model', String(100), nullable=False),
    Column('embedding_dimensions', Integer, nullable=False),
    Column('created_at', DateTime(timezone=True), nullable=False))

files = Table('pl_files', metadata,
    Column('id', String(32), primary_key=True),
    Column('project_id', ForeignKey('pl_projects.id'), nullable=False, index=True),
    Column('filename', String(255), nullable=False),
    Column('content_hash', String(64), nullable=False),
    Column('r2_key', String(500), nullable=False),
    Column('status', String(20), nullable=False),
    Column('error', Text),
    Column('size_bytes', Integer, nullable=False),
    Column('total_pages', Integer, nullable=False, default=0),
    Column('chunk_count', Integer, nullable=False, default=0),
    Column('paper', json_type),
    Column('summary', json_type),
    Column('created_at', DateTime(timezone=True), nullable=False),
    UniqueConstraint('project_id', 'content_hash'))

messages = Table('pl_messages', metadata,
    Column('id', String(32), primary_key=True),
    Column('project_id', ForeignKey('pl_projects.id'), nullable=False, index=True),
    Column('role', String(12), nullable=False),
    Column('content', Text, nullable=False),
    Column('payload', json_type),
    Column('created_at', DateTime(timezone=True), nullable=False))


class Repository:
    def __init__(self, engine=None):
        self.engine = engine if engine is not None else create_engine(
            database_url(), pool_pre_ping=True, pool_size=5, max_overflow=5,
            connect_args={'connect_timeout': 10}, hide_parameters=True)

    def initialize(self):
        metadata.create_all(self.engine)

    def user(self, subject, email, name):
        # Google subject, rather than mutable email, is the permanent identity.
        identity = 'google:' + subject
        with self.engine.begin() as conn:
            if conn.dialect.name == 'postgresql':
                from sqlalchemy.dialects.postgresql import insert
                conn.execute(insert(users).values(id=uid(), identity=identity, email=email,
                    name=name, created_at=now()).on_conflict_do_update(
                        index_elements=['identity'], set_={'email': email, 'name': name}))
            else:  # Isolated test database only; runtime configuration requires PostgreSQL.
                row = conn.execute(select(users).where(users.c.identity == identity)).mappings().first()
                if row is None:
                    conn.execute(users.insert().values(id=uid(), identity=identity, email=email,
                                                        name=name, created_at=now()))
            return dict(conn.execute(select(users).where(users.c.identity == identity)).mappings().one())

    def _project(self, conn, user_id, project_id, lock=False):
        stmt = select(projects).where(projects.c.id == project_id, projects.c.user_id == user_id)
        if lock:
            stmt = stmt.with_for_update()
        row = conn.execute(stmt).mappings().first()
        if row is None:
            raise ValueError('Project unavailable or access denied.')
        return dict(row)

    def project(self, user_id, project_id):
        with self.engine.connect() as conn:
            return self._project(conn, user_id, project_id)

    def list_projects(self, user_id):
        with self.engine.connect() as conn:
            return [dict(r) for r in conn.execute(select(projects).where(
                projects.c.user_id == user_id).order_by(projects.c.created_at.desc())).mappings()]

    def create_project(self, user_id, name, model, dimensions):
        name = name.strip()
        if not name or len(name) > 120:
            raise ValueError('Use a project name between 1 and 120 characters.')
        with self.engine.begin() as conn:
            conn.execute(select(users).where(users.c.id == user_id).with_for_update()).one()
            count = conn.scalar(select(func.count()).select_from(projects).where(projects.c.user_id == user_id))
            if count >= integer('MAX_PROJECTS_PER_USER', 20):
                raise ValueError('Project limit reached.')
            project_id = uid()
            conn.execute(projects.insert().values(id=project_id, user_id=user_id, name=name,
                namespace='project-' + project_id, embedding_model=model,
                embedding_dimensions=dimensions, created_at=now()))
            return project_id

    def list_files(self, user_id, project_id):
        with self.engine.connect() as conn:
            self._project(conn, user_id, project_id)
            return [dict(r) for r in conn.execute(select(files).where(files.c.project_id == project_id)
                .order_by(files.c.created_at)).mappings()]

    def file(self, user_id, project_id, file_id):
        with self.engine.connect() as conn:
            self._project(conn, user_id, project_id)
            row = conn.execute(select(files).where(files.c.id == file_id,
                files.c.project_id == project_id)).mappings().first()
            if row is None:
                raise ValueError('File unavailable or access denied.')
            return dict(row)

    def reserve_file(self, user_id, project_id, filename, digest, size):
        with self.engine.begin() as conn:
            self._project(conn, user_id, project_id, lock=True)
            duplicate = conn.execute(select(files).where(files.c.project_id == project_id,
                files.c.content_hash == digest)).mappings().first()
            if duplicate:
                raise ValueError('This PDF is already in the project. Remove its existing record before retrying.')
            count = conn.scalar(select(func.count()).select_from(files).where(files.c.project_id == project_id))
            if count >= integer('MAX_FILES_PER_PROJECT', 30):
                raise ValueError('This project has reached its file limit.')
            file_id = uid()
            key = f'users/{user_id}/projects/{project_id}/{file_id}.pdf'
            conn.execute(files.insert().values(id=file_id, project_id=project_id,
                filename=filename, content_hash=digest, size_bytes=size, r2_key=key,
                status='processing', total_pages=0, chunk_count=0, created_at=now()))
            return file_id, key

    def update_file(self, user_id, project_id, file_id, **values):
        if set(values) - {'status', 'error', 'total_pages', 'chunk_count', 'paper', 'summary'}:
            raise ValueError('Invalid file update.')
        with self.engine.begin() as conn:
            self._project(conn, user_id, project_id, lock=True)
            result = conn.execute(files.update().where(files.c.id == file_id,
                files.c.project_id == project_id).values(**values))
            if not result.rowcount:
                raise ValueError('File unavailable or access denied.')

    def remove_file_record(self, user_id, project_id, file_id):
        with self.engine.begin() as conn:
            self._project(conn, user_id, project_id, lock=True)
            conn.execute(files.delete().where(files.c.id == file_id, files.c.project_id == project_id))

    def history(self, user_id, project_id, limit=100):
        with self.engine.connect() as conn:
            self._project(conn, user_id, project_id)
            rows = conn.execute(select(messages).where(messages.c.project_id == project_id)
                .order_by(messages.c.created_at.desc(), messages.c.id.desc()).limit(limit)).mappings()
            return list(reversed([dict(r) for r in rows]))

    def save_exchange(self, user_id, project_id, question, payload):
        with self.engine.begin() as conn:
            self._project(conn, user_id, project_id, lock=True)
            count = conn.scalar(select(func.count()).select_from(messages).where(messages.c.project_id == project_id))
            if count + 2 > integer('MAX_MESSAGES_PER_PROJECT', 1000):
                raise ValueError('Chat history limit reached for this project.')
            answer_text = '\n\n'.join([payload['result']['text']] +
                [claim['text'] for claim in payload['result']['claims']])
            for role, content, data in [('user', question, None),
                                       ('assistant', answer_text, payload)]:
                conn.execute(messages.insert().values(id=uid(), project_id=project_id,
                    role=role, content=content, payload=data, created_at=now()))
