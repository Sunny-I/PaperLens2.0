"""Summary citations resolve to server-held text rather than copied model quotes."""
import json
import unittest
from unittest.mock import Mock

from pipeline.summarizer import PaperSummarizer, reference_batch


def fields(**updates):
    return dict({'research_problem': None, 'methodology': None,
                 'datasets_and_benchmarks': None, 'key_findings': [], 'limitations': []}, **updates)


class SummaryReferenceTests(unittest.TestCase):
    def test_original_accented_pdf_text_is_attached_to_summary(self):
        text = 'Tolga ¸Sakar describes a method for improving RAG efﬁciency.'
        paper = {'filename': 'paper.pdf', 'raw_text': text, 'pages': [{'page': 1, 'text': text}]}
        client = Mock()
        def respond(model, messages, schema, validator):
            part = schema.model_validate(fields(methodology={'text': 'The paper improves RAG efficiency.',
                'evidence': [{'page': 1, 'evidence_id': 1}]}))
            validator(part)
            return part
        client.structured.side_effect = respond
        result = PaperSummarizer().summarize(paper, client, 'test', evidence_ids=True)
        self.assertEqual(result.mode, 'generated')
        self.assertEqual(result.summary.evidence['methodology'][0].quote, text)

    def test_unknown_reference_is_rejected(self):
        paper = {'filename': 'paper.pdf', 'raw_text': '', 'pages': [{'page': 1, 'text': 'Actual source passage.'}]}
        client = Mock()
        def respond(model, messages, schema, validator):
            part = schema.model_validate(fields(methodology={'text': 'Claim',
                'evidence': [{'page': 99, 'evidence_id': 1}]}))
            validator(part)
            return part
        client.structured.side_effect = respond
        result = PaperSummarizer().summarize(paper, client, 'test', evidence_ids=True)
        self.assertEqual(result.mode, 'extractive')
        self.assertIn('failed', result.warning)

    def test_all_pages_are_processed_and_late_finding_keeps_correct_page(self):
        seen = set()
        client = Mock()
        def respond(model, messages, schema, validator):
            pages = json.loads(messages[1]['content'])['pages']
            seen.update(p['page'] for p in pages)
            last = next((p for p in pages if p['page'] == 31), None)
            part = schema.model_validate(fields(key_findings=[{'text': 'Late-page finding.', 'evidence': [
                {'page': 31, 'evidence_id': last['evidence'][0]['evidence_id']}]}] if last else []))
            validator(part)
            return part
        client.structured.side_effect = respond
        pages = [{'page': i, 'text': 'Source evidence with useful context. ' * 100} for i in range(1, 32)]
        result = PaperSummarizer().summarize({'filename': 'long.pdf', 'raw_text': '', 'pages': pages},
                                           client, 'test', evidence_ids=True)
        self.assertEqual(result.mode, 'generated')
        self.assertEqual(seen, set(range(1, 32)))
        self.assertEqual(result.summary.evidence['key_findings'][0].page, 31)

    def test_same_page_pieces_get_distinct_evidence_ids(self):
        sources, evidence = reference_batch([{'page': 2, 'text': 'First source piece.'},
                                             {'page': 2, 'text': 'Second source piece.'}])
        self.assertEqual(evidence[(2, 1)], 'First source piece.')
        self.assertEqual(evidence[(2, 2)], 'Second source piece.')


if __name__ == '__main__':
    unittest.main()
