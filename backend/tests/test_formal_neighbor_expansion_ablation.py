"""Tests for the post-hoc formal neighbor-expansion ablation."""

from __future__ import annotations

import json
from pathlib import Path

from apps.evaluations.formal_neighbor_expansion_ablation import (
    build_formal_neighbor_expansion_ablation,
)


def _qra(qra_id: str, domain: str) -> dict[str, object]:
    return {
        "id": qra_id,
        "domain": domain,
        "question_type": "Factual",
        "question": f"Question {qra_id}",
        "keypoints": [{"id": "K1", "reference_ids": ["R1"]}],
        "references": [
            {
                "reference_id": "R1",
                "raw_source_path": f"data/{domain}/source.txt",
                "quote": "exact evidence",
            }
        ],
    }


def test_neighbor_ablation_reuses_frozen_selected_chunks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    domains = [
        "market_validation",
        "funding_preparation",
        "legal_form_selection",
    ]
    qras = [_qra(f"Q{index}", domain) for index, domain in enumerate(domains)]
    results = []
    for qra in qras:
        domain = qra["domain"]
        hits = [
            {
                "rank": rank,
                "dense_rank": rank,
                "score": 1.0 / rank,
                "chunk_id": f"{domain}/chunk-{rank}",
                "source_path": (
                    f"{domain}/source.txt"
                    if rank == 1
                    else f"{domain}/other-{rank}.txt"
                ),
                "chunk_index": rank,
                "text": "wrong chunk",
            }
            for rank in range(1, 21)
        ]
        results.append(
            {
                "status": "success",
                "qra_id": qra["id"],
                "dense_candidates": hits,
                "selected_chunks": hits[:5],
            }
        )
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "status": "completed",
                "rag_index": {
                    "metadata_sha256": "metadata",
                    "embeddings_sha256": "embeddings",
                },
                "results": results,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "apps.evaluations.formal_neighbor_expansion_ablation.snapshot_rag_index",
        lambda: {
            "metadata_sha256": "metadata",
            "embeddings_sha256": "embeddings",
        },
    )

    def fake_expand(hits, **kwargs):  # noqa: ANN001, ANN003
        expanded = [dict(hit) for hit in hits]
        expanded[0]["text"] += " exact evidence"
        expanded[0]["expanded_neighbor_chunk_ids"] = ["neighbor"]
        return expanded, 200, 1

    paths = build_formal_neighbor_expansion_ablation(
        qra_items=qras,
        source_report_path=source,
        output_dir=tmp_path,
        report_id="NEIGHBOR-TEST",
        expansion_fn=fake_expand,
    )

    report = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert report["summaries"]["rerank_neighbor_2"]["overall"][
        "macro_reference_recall_at_k"
    ] == 1.0
    assert report["acceptance"]["all_gates_passed"] is True
    assert "Neighbor Expansion" in paths["markdown"].read_text(encoding="utf-8")
