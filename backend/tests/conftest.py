"""Exclude optional corpus-integrity tests when external artifacts are absent."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ARTIFACT_TESTS = {'test_formal_configuration_candidates.py': 'docs/evaluation/formal_dataset',
 'test_formal_configuration_freeze.py': 'docs/evaluation/formal_dataset',
 'test_formal_document_candidates.py': 'docs/evaluation/formal_dataset',
 'test_formal_qra_candidates.py': 'docs/evaluation/formal_dataset',
 'test_formal_qra_freezing.py': 'docs/evaluation/formal_dataset',
 'test_formal_qra_review_revisions.py': 'docs/evaluation/formal_dataset',
 'test_formal_qra_semantic_review.py': 'docs/evaluation/formal_dataset',
 'test_formal_upstream_revisions.py': 'docs/evaluation/formal_dataset',
 'test_legal_corpus.py': 'knowledge_base/legal_form_selection/primary_law',
 'test_optimization_configuration_candidates.py': 'docs/evaluation/formal_dataset',
 'test_optimization_configuration_freeze.py': 'docs/evaluation/optimization_dataset',
 'test_optimization_document_candidates.py': 'docs/evaluation/optimization_dataset',
 'test_optimization_document_freeze.py': 'docs/evaluation/optimization_dataset',
 'test_optimization_qra_candidates.py': 'docs/evaluation/formal_dataset',
 'test_optimization_qra_freeze.py': 'docs/evaluation/optimization_dataset',
 'test_primary_law_configuration_curation.py': 'docs/evaluation/formal_dataset',
 'test_primary_law_configuration_freezing.py': 'docs/evaluation/formal_dataset',
 'test_primary_law_configuration_revisions.py': 'docs/evaluation/formal_dataset',
 'test_primary_law_document_candidates.py': 'docs/evaluation/formal_dataset',
 'test_primary_law_document_freezing.py': 'docs/evaluation/formal_dataset',
 'test_primary_law_qra_revisions.py': 'docs/evaluation/formal_dataset',
 'test_primary_law_review_preparation.py': 'docs/evaluation/formal_dataset'}

collect_ignore = [
    name for name, required_path in EXTERNAL_ARTIFACT_TESTS.items()
    if not (REPO_ROOT / required_path).exists()
]
