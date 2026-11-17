"""Safe structured-output validation feedback without provider response bodies."""


class CitationValidationError(ValueError):
    def __init__(self, source_id, issue):
        self.source_id = source_id
        self.issue = issue
        if issue == 'source':
            self.repair_hint = f'Source ID {source_id} does not exist. Use only supplied source IDs.'
        elif issue == 'evidence':
            self.repair_hint = (f'The evidence ID is not present in source {source_id}. '
                                'Use only the supplied source_id/evidence_id pairs.')
        else:
            self.repair_hint = (f'The quote for source ID {source_id} was not a valid exact passage. '
                'Copy one contiguous passage from that source verbatim, preserving punctuation and spelling. '
                'Do not join separate passages or paraphrase inside a quote. '
                'Drop any claim for which you cannot provide a matching quote.')
        super().__init__('Unknown source ID in generated answer.' if issue == 'source'
                         else 'Generated citation does not match the source text.')


class StructuredOutputError(ValueError):
    def __init__(self, stage):
        self.stage = stage
        label = 'citation validation' if stage == 'citations' else 'structured-output validation'
        super().__init__(f'OpenAI {label} failed twice.')
