"""Regression checks for the full formal QRA v0.4 semantic review gate."""

import json
from pathlib import Path

from apps.evaluations.rageval_artifacts import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[2]
QRA_ROOT = REPO_ROOT / "docs/evaluation/formal_dataset/qra"
CANDIDATE_MANIFEST_PATH = (
    QRA_ROOT / "formal_candidates/generation_manifest_v0.4.json"
)
PREDECESSOR_MANIFEST_PATH = (
    QRA_ROOT / "formal_candidates/generation_manifest_v0.3.json"
)
REVISION_MANIFEST_PATH = QRA_ROOT / "primary_law_qra_revision_manifest_v0.4.json"
AGENT_REVIEW_PATH = (
    QRA_ROOT / "reviews/round_3/formal_qra_agent_semantic_review_v0.4.json"
)
OWNER_CONFIRMATION_PATH = (
    QRA_ROOT / "reviews/round_3/formal_qra_project_owner_confirmation_v0.4.json"
)

SEMANTIC_FIELDS = (
    "question",
    "answerable",
    "answer",
    "keypoints",
    "references",
    "answer_reference_alignment",
    "document_ids",
    "question_type",
    "domain",
    "difficulty",
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_agent_review_covers_all_items_without_false_human_claims() -> None:
    review = _read_json(AGENT_REVIEW_PATH)
    confirmation = _read_json(OWNER_CONFIRMATION_PATH)

    assert review["status"] == "agent_semantic_review_passed"
    assert review["qra_count"] == 60
    assert review["revised_item_count"] == 17
    assert review["carried_item_count"] == 43
    assert review["mechanically_valid_count"] == 60
    assert review["agent_semantically_reviewed_count"] == 60
    assert review["unresolved_blocker_count"] == 0
    assert review["human_review_claimed"] is False
    assert review["legal_expert_review_claimed"] is False
    assert review["agent_review_is_human_review"] is False
    assert len(review["items"]) == 60
    assert {item["agent_result"] for item in review["items"]} == {"pass"}
    assert review["candidate_manifest"]["sha256"] == sha256_file(
        CANDIDATE_MANIFEST_PATH
    )

    assert confirmation["manual_review_reported"] is True
    assert confirmation["reported_issue_count"] == 0
    assert confirmation["freeze_condition_satisfied"] is True
    assert (
        confirmation["final_post_correction_human_line_by_line_review_claimed"]
        is False
    )
    assert confirmation["independent_second_human_review_claimed"] is False
    assert confirmation["legal_expert_review_claimed"] is False
    assert confirmation["agent_review"]["sha256"] == sha256_file(
        AGENT_REVIEW_PATH
    )


def test_43_carried_qras_have_no_semantic_drift_from_v0_3() -> None:
    current_manifest = _read_json(CANDIDATE_MANIFEST_PATH)
    predecessor_manifest = _read_json(PREDECESSOR_MANIFEST_PATH)
    revision_manifest = _read_json(REVISION_MANIFEST_PATH)
    revised_configuration_ids = {
        item["configuration_id"] for item in revision_manifest["revisions"]
    }
    predecessors = {
        item["qra_id"]: _read_json(REPO_ROOT / item["path"])
        for item in predecessor_manifest["qras"]
    }

    carried = 0
    for declaration in current_manifest["qras"]:
        if declaration["configuration_id"] in revised_configuration_ids:
            continue
        carried += 1
        current = _read_json(REPO_ROOT / declaration["path"])
        predecessor = predecessors[current["id"]]
        assert {
            field: current[field] for field in SEMANTIC_FIELDS
        } == {
            field: predecessor[field] for field in SEMANTIC_FIELDS
        }
    assert carried == 43


def test_resolved_primary_law_evidence_gaps_do_not_regress() -> None:
    manifest = _read_json(CANDIDATE_MANIFEST_PATH)
    qras = {
        item["qra_id"]: _read_json(REPO_ROOT / item["path"])
        for item in manifest["qras"]
    }

    legal_boundary = qras["FTLFS-F-002"]
    assert "keine vollständige Liste aller freien Berufe" not in legal_boundary[
        "answer"
    ]
    assert "das Unterrichtswesen" in legal_boundary["answer"]

    partg = qras["FTLFS-MHR-002"]
    assert "Partnerschaftsvertrag" in partg["answer"]
    assert "§ 124 HGB" in partg["answer"]
    assert "mit dem Auftrag befassten Partner" in partg["answer"]

    bookkeeping = qras["FTLFS-II-002"]
    for term in ("Umsatzerlöse", "Jahresüberschuss", "Gesamtumsatz", "Gewinn"):
        assert term in bookkeeping["answer"]

    grant = qras["FTFP-TS-002"]
    assert "Kenntnisse und Fähigkeiten" in grant["answer"]
    assert "persönliche Eignung" not in grant["answer"]
