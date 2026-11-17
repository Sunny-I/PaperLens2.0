"""Paper identity questions must use front matter, not similar references."""
import unittest
from unittest.mock import Mock

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from workspace.database import Repository
from workspace.service import WorkspaceService, is_paper_metadata_question
from test_workspace import FakeEmbeddings, FakeStorage, FakeVectors


class MetadataRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine('sqlite://', connect_args={'check_same_thread': False}, poolclass=StaticPool)
        self.addCleanup(self.engine.dispose)
        self.repo = Repository(self.engine)
        self.repo.initialize()
        self.owner = self.repo.user('metadata-owner', 'owner@example.test', 'Owner')['id']
        self.project = self.repo.create_project(self.owner, 'Papers', 'model', 2)
        self.vectors = FakeVectors()
        self.service = WorkspaceService(self.repo, FakeStorage(), self.vectors, FakeEmbeddings)

    def add_paper(self, name, author, project=None):
        project = project or self.project
        file_id, key = self.repo.reserve_file(self.owner, project, name, name.ljust(64, 'x'), 100)
        paper = {'pages': [{'page': 1, 'text': f'{name}\n{author}\nAbstract: A research study.'},
                           {'page': 2, 'text': 'Introduction: We study a research method.'},
                           {'page': 25, 'text': 'References\nOther Writer. Similar paper. 2023.'}],
                 'chunks': [{'chunk_id': file_id + '-ref', 'doc_id': file_id, 'filename': name,
                             'page': 25, 'text': 'References Other Writer. Similar paper. 2023.'}]}
        self.repo.update_file(self.owner, project, file_id, status='ready', paper=paper)
        return file_id

    def test_typo_author_question_uses_headers_and_not_bibliography(self):
        file_id = self.add_paper('study.pdf', 'Alice Researcher and Bob Scientist')
        self.vectors.query = Mock(side_effect=AssertionError('Metadata should not use semantic search'))
        result = self.service.ask(self.owner, self.project, 'what is the names of authers', generate=False)
        self.assertEqual(result['configuration']['retrieval_strategy'], 'front_matter')
        self.assertIn('Alice Researcher', result['hits'][0]['text'])
        self.assertEqual({hit['doc_id'] for hit in result['hits']}, {file_id})
        self.assertNotIn(25, [hit['page'] for hit in result['hits']])
        self.vectors.query.assert_not_called()

    def test_more_than_six_files_all_receive_front_page_coverage(self):
        file_ids = {self.add_paper(f'paper-{i}.pdf', f'Author {i}') for i in range(8)}
        result = self.service.ask(self.owner, self.project, 'List the authors of these papers', generate=False)
        self.assertEqual({hit['doc_id'] for hit in result['hits']}, file_ids)

    def test_selection_and_project_boundaries_apply_to_header_retrieval(self):
        chosen = self.add_paper('chosen.pdf', 'Chosen Author')
        self.add_paper('unselected.pdf', 'Unselected Author')
        other = self.repo.create_project(self.owner, 'Other', 'model', 2)
        self.add_paper('private-other.pdf', 'Other Project Author', other)
        result = self.service.ask(self.owner, self.project, 'Who wrote these papers?',
                                  selected_ids=[chosen], generate=False)
        self.assertEqual({hit['doc_id'] for hit in result['hits']}, {chosen})

    def test_scientific_author_question_keeps_semantic_search(self):
        self.add_paper('study.pdf', 'Alice Researcher')
        self.service.ask(self.owner, self.project, 'What do the authors propose?', generate=False)
        self.assertEqual(len(self.vectors.calls), 1)

    def test_intent_detection_does_not_treat_method_questions_as_identity(self):
        for question in ['who wrote this paper?', 'What are the author names?', 'authors',
                         'What is the title of this paper?', 'what is the names of authers']:
            self.assertTrue(is_paper_metadata_question(question), question)
        for question in ['What do the authors conclude?', 'Explain the method',
                         'Why did the authors choose this dataset?']:
            self.assertFalse(is_paper_metadata_question(question), question)


if __name__ == '__main__':
    unittest.main()
