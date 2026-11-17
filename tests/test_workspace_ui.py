"""UI integration with a real relational repository and mocked cloud services."""
import unittest
from pathlib import Path
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from workspace.database import Repository
from workspace.service import WorkspaceService
from test_workspace import FakeStorage, FakeVectors, FakeEmbeddings, pdf

APP = str(Path(__file__).resolve().parents[1] / 'app.py')


class Identity(dict):
    @property
    def is_logged_in(self):
        return bool(self.get('sub'))


class WorkspaceUITests(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()
        self.engine = create_engine('sqlite://', connect_args={'check_same_thread': False}, poolclass=StaticPool)
        self.repo = Repository(self.engine)
        self.repo.initialize()
        self.service = WorkspaceService(self.repo, FakeStorage(), FakeVectors(), FakeEmbeddings)
        self.addCleanup(self.engine.dispose)
        self.addCleanup(st.cache_resource.clear)

    def patches(self, identity):
        return [patch('workspace.settings.missing_configuration', return_value=[]),
                patch('streamlit.user', identity),
                patch('workspace.database.Repository', return_value=self.repo),
                patch('workspace.service.WorkspaceService', return_value=self.service)]

    def test_setup_screen_is_actionable_and_does_not_access_database(self):
        with patch('workspace.settings.missing_configuration', return_value=['DATABASE_URL']), \
             patch('workspace.database.Repository') as constructor:
            app = AppTest.from_file(APP).run(timeout=30)
            self.assertFalse(app.exception)
            self.assertIn('almost ready', app.subheader[0].value)
            constructor.assert_not_called()

    def test_logged_out_user_sees_google_login_and_no_private_data(self):
        patches = self.patches(Identity())
        with patches[0], patches[1], patches[2] as constructor, patches[3]:
            app = AppTest.from_file(APP).run(timeout=30)
            self.assertFalse(app.exception)
            self.assertTrue(any(b.label == 'Continue with Google' for b in app.button))
            self.assertFalse(app.tabs)
            constructor.assert_not_called()

    def test_create_project_and_reload_chat_from_database(self):
        patches = self.patches(Identity(sub='ui-user', name='Researcher', email='ui@example.test'))
        with patches[0], patches[1], patches[2], patches[3]:
            app = AppTest.from_file(APP).run(timeout=30)
            self.assertFalse(app.exception)
            next(t for t in app.text_input if t.label == 'Project name').set_value('My research')
            next(b for b in app.button if b.label == '＋ Create project').click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual([tab.label for tab in app.tabs], ['Files', 'Chat', 'Summaries'])
            user = self.repo.user('ui-user', 'ui@example.test', 'Researcher')['id']
            project = self.repo.list_projects(user)[0]['id']
            self.service.upload(user, project, 'study.pdf', pdf())
            self.service.ask(user, project, 'What accuracy?', generate=False)
            # A fresh Streamlit session must reload project files and saved evidence.
            fresh = AppTest.from_file(APP).run(timeout=30)
            self.assertFalse(fresh.exception)
            self.assertTrue(any(m.value == '1' for m in fresh.metric))
            self.assertEqual(len(fresh.chat_message), 2)
            self.assertTrue(any('study.pdf' in e.label and 'page 1' in e.label for e in fresh.expander))


if __name__ == '__main__':
    unittest.main()
