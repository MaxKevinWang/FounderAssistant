"""Tests for the blind OpenAI API keypoint-Judge pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from apps.evaluations import openai_judge as judge


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _source_run_fixture(tmp_path: Path, *, repeats: int = 1) -> Path:
    path = tmp_path / "source_run.json"
    qra = {
        "id": "TEST-F-001",
        "domain": "market_validation",
        "question_type": "Factual",
        "language": "de",
        "question": "Welche Punkte sind richtig?",
        "answerable": True,
        "reference_answer": "Punkt eins und Punkt zwei sind richtig.",
        "keypoints": [
            {"id": "K1", "text": "Punkt eins", "required": True},
            {"id": "K2", "text": "Punkt zwei", "required": True},
            {"id": "K3", "text": "Punkt drei", "required": True},
        ],
        "references": [
            {"reference_id": "R1", "quote": "Beleg für Punkt eins."},
            {"reference_id": "R2", "quote": "Beleg für Punkt zwei."},
        ],
    }
    results = []
    sequence = 0
    for repeat in range(1, repeats + 1):
        for mode, answer in (("baseline", "Antwort A"), ("rag", "Antwort B")):
            sequence += 1
            results.append(
                {
                    "sequence": sequence,
                    "qra_id": qra["id"],
                    "domain": qra["domain"],
                    "question_type": qra["question_type"],
                    "model": "test:model",
                    "model_type": "dense",
                    "mode": mode,
                    "repeat": repeat,
                    "generation": {
                        "status": "success",
                        "response": answer,
                    },
                }
            )
    _write_json(
        path,
        {
            "run_id": "source-test-run",
            "status": "completed",
            "dataset": {
                "dataset_id": "test-dataset",
                "manifest_sha256": "dataset-hash",
                "items": [qra],
            },
            "results": results,
        },
    )
    return path


def _prompt_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "judge_prompt.json"
    _write_json(
        path,
        {
            "prompt_id": "test-judge",
            "version": "0.1",
            "status": "development",
            "purpose": "test",
            "system_prompt": "Evaluate supplied keypoints.",
            "user_prompt_template": "{evaluation_item_json}",
        },
    )
    return path


def _decision_for(judge_input: dict) -> dict:
    rag_answer = judge_input["candidate_answer"] == "Antwort B"
    labels = {
        "K1": "covered",
        "K2": "covered" if rag_answer else "contradicted",
        "K3": "neither",
    }
    decision = {
        "keypoint_evaluations": [
            {
                "keypoint_id": item["id"],
                "label": labels[item["id"]],
                "rationale": f"{item['id']} rationale",
                "answer_evidence": (
                    judge_input["candidate_answer"]
                    if labels[item["id"]] in {"covered", "contradicted"}
                    else ""
                ),
            }
            for item in judge_input["keypoints"]
        ],
        "unsupported_claims": [],
        "overall_assessment": "Test assessment.",
    }
    return {
        "response_id": "response-test",
        "response_model": "gpt-test",
        "raw_output_text": json.dumps(decision),
        "decision": decision,
        "usage": {
            "input_tokens": 100,
            "cached_input_tokens": 0,
            "output_tokens": 30,
            "reasoning_tokens": 10,
            "total_tokens": 130,
        },
    }


def test_prepare_candidates_hides_model_and_condition_labels(
    tmp_path: Path,
) -> None:
    source = judge.load_generation_run(_source_run_fixture(tmp_path))

    selection = judge.prepare_blinded_candidates(
        source,
        tested_models=["test:model"],
        qra_ids=["TEST-F-001"],
        seed=42,
    )

    assert selection["candidate_count"] == 2
    for candidate in selection["candidates"]:
        assert "tested_model" not in candidate["judge_input"]
        assert "mode" not in candidate["judge_input"]
        assert set(candidate["judge_input"]) == {
            "qra_id",
            "language",
            "question",
            "answerable",
            "reference_answer",
            "keypoints",
            "reference_evidence",
            "candidate_answer",
        }


def test_prepare_and_pair_repeated_candidates(tmp_path: Path) -> None:
    source = judge.load_generation_run(_source_run_fixture(tmp_path, repeats=2))

    selection = judge.prepare_blinded_candidates(source, seed=42)

    assert selection["candidate_count"] == 4
    assert {candidate["repeat"] for candidate in selection["candidates"]} == {1, 2}

    scored = []
    for candidate in selection["candidates"]:
        scored.append(
            {
                **{
                    key: candidate[key]
                    for key in (
                        "candidate_id",
                        "source_sequence",
                        "qra_id",
                        "domain",
                        "question_type",
                        "tested_model",
                        "tested_model_type",
                        "mode",
                        "repeat",
                    )
                },
                "judge": {
                    "status": "success",
                    "metrics": {
                        "completeness": 1.0 if candidate["mode"] == "rag" else 0.5,
                        "hallucination": 0.0,
                        "irrelevancy": 0.0 if candidate["mode"] == "rag" else 0.5,
                        "unsupported_claim_count": None,
                    },
                },
            }
        )

    comparison = judge.build_paired_comparison(scored)

    assert len(comparison["pairs"]) == 2
    assert {pair["repeat"] for pair in comparison["pairs"]} == {1, 2}
    assert comparison["summary"][0]["paired_qra_count"] == 1
    assert comparison["summary"][0]["paired_observation_count"] == 2
    assert comparison["summary"][0]["mean_rag_minus_baseline_completeness"] == 0.5


def test_openai_callable_uses_responses_structured_outputs() -> None:
    captured: dict = {}

    class FakeResponses:
        def parse(self, **kwargs):
            captured.update(kwargs)
            parsed = kwargs["text_format"].model_validate(
                {
                    "keypoint_evaluations": {
                        "K1": {
                            "label": "covered",
                            "rationale": "Supported.",
                            "answer_evidence": "",
                        }
                    },
                    "unsupported_claims": [],
                    "overall_assessment": "Supported.",
                }
            )
            return SimpleNamespace(
                id="resp-1",
                model="gpt-5.4-mini",
                output_parsed=parsed,
                output_text=parsed.model_dump_json(),
                usage=SimpleNamespace(
                    input_tokens=20,
                    output_tokens=10,
                    total_tokens=30,
                    input_tokens_details=SimpleNamespace(cached_tokens=2),
                    output_tokens_details=SimpleNamespace(reasoning_tokens=3),
                ),
            )

    api_judge = judge.OpenAIKeypointJudge(
        prompt={
            "system_prompt": "Evaluate.",
            "user_prompt_template": "{evaluation_item_json}",
        },
        model="gpt-5.4-mini",
        reasoning_effort="low",
        max_output_tokens=1000,
        client=SimpleNamespace(responses=FakeResponses()),
    )

    result = api_judge(
        {
            "keypoints": [{"id": "K1", "text": "One"}],
            "candidate_answer": "One",
        }
    )

    assert captured["model"] == "gpt-5.4-mini"
    assert captured["store"] is False
    assert captured["reasoning"] == {"effort": "low"}
    schema = captured["text_format"].model_json_schema()
    keypoint_schema = schema["$defs"]["RequiredKeypointEvaluations"]
    assert keypoint_schema["required"] == ["K1"]
    assert keypoint_schema["additionalProperties"] is False
    assert result["usage"]["cached_input_tokens"] == 2
    assert result["usage"]["reasoning_tokens"] == 3


def test_api_callable_configures_custom_responses_base_url(monkeypatch) -> None:
    openai_constructor = Mock(
        return_value=SimpleNamespace(responses=SimpleNamespace())
    )
    monkeypatch.setattr("openai.OpenAI", openai_constructor)

    judge.OpenAIKeypointJudge(
        prompt={
            "system_prompt": "Evaluate.",
            "user_prompt_template": "{evaluation_item_json}",
        },
        model="deepseek-v4-flash",
        api_key="deepseek-test-key",
        base_url="https://api.deepseek.example",
        timeout_seconds=360,
        max_retries=3,
    )

    openai_constructor.assert_called_once_with(
        api_key="deepseek-test-key",
        base_url="https://api.deepseek.example",
        timeout=360,
        max_retries=3,
    )


def test_reference_isolated_prompt_hides_reference_text_from_model() -> None:
    captured: dict = {}

    class FakeResponses:
        def parse(self, **kwargs):
            captured.update(kwargs)
            parsed = kwargs["text_format"].model_validate(
                {
                    "keypoint_evaluations": {
                        "K1": {
                            "label": "neither",
                            "rationale": "The deadline is absent.",
                            "answer_evidence": "",
                        }
                    },
                    "unsupported_claims": [],
                    "overall_assessment": "The composite keypoint is incomplete.",
                }
            )
            return SimpleNamespace(
                id="resp-isolated",
                model="gpt-test",
                output_parsed=parsed,
                output_text=parsed.model_dump_json(),
                usage=None,
            )

    api_judge = judge.OpenAIKeypointJudge(
        prompt={
            "system_prompt": "Evaluate every required element.",
            "user_prompt_template": "{evaluation_item_json}",
            "judge_input_policy": "keypoint_only_reference_isolated",
            "judge_input_fields": [
                "qra_id",
                "language",
                "question",
                "answerable",
                "keypoints",
                "candidate_answer",
            ],
            "unsupported_claims_evaluation": "not_evaluated",
        },
        model="gpt-test",
        client=SimpleNamespace(responses=FakeResponses()),
    )

    result = api_judge(
        {
            "qra_id": "LFS-TS-002",
            "language": "de",
            "question": "Welche Frist gilt?",
            "answerable": True,
            "keypoints": [
                {
                    "id": "K1",
                    "text": "Fragebogen innerhalb eines Monats einreichen.",
                }
            ],
            "candidate_answer": "Den Fragebogen nach der Anmeldung ausf眉llen.",
            "reference_answer": "Innerhalb eines Monats einreichen.",
            "reference_evidence": [
                {"reference_id": "R1", "quote": "innerhalb eines Monats"}
            ],
        }
    )

    visible_input = captured["input"][1]["content"]
    assert "candidate_answer" in visible_input
    assert "reference_answer" not in visible_input
    assert "reference_evidence" not in visible_input
    assert (
        '"reference_answer": "Innerhalb eines Monats einreichen."' not in visible_input
    )
    assert result["unsupported_claims_evaluation"] == "not_evaluated"


def test_default_prompt_is_segmented_reference_isolated_v05() -> None:
    prompt = judge.load_judge_prompt()

    assert prompt["version"] == "0.5"
    assert prompt["judge_input_policy"] == "keypoint_segmented_reference_isolated"
    assert prompt["answer_evidence_format"] == "segment_id_list"
    assert prompt["unsupported_claims_evaluation"] == "not_evaluated"


def test_reference_isolated_score_does_not_report_zero_unsupported_claims() -> None:
    judge_input = {
        "keypoints": [{"id": "K1", "text": "Punkt eins"}],
        "candidate_answer": "Punkt eins ist enthalten.",
    }
    decision = {
        "keypoint_evaluations": {
            "K1": {
                "label": "covered",
                "rationale": "Punkt eins is explicit.",
                "answer_evidence": "Punkt eins",
            }
        },
        "unsupported_claims": [],
        "overall_assessment": "Covered.",
    }

    result = judge._validate_and_score_decision(
        decision,
        judge_input,
        unsupported_claims_evaluation="not_evaluated",
    )

    assert result["unsupported_claims_evaluation"] == "not_evaluated"
    assert result["metrics"]["unsupported_claim_count"] is None


def test_invalid_dynamic_keypoint_ids_are_rejected() -> None:
    decision = {
        "keypoint_evaluations": [
            {
                "keypoint_id": "WRONG",
                "label": "covered",
                "rationale": "Wrong ID.",
                "answer_evidence": "",
            }
        ],
        "unsupported_claims": [],
        "overall_assessment": "",
    }
    judge_input = {
        "keypoints": [{"id": "K1", "text": "One"}],
        "candidate_answer": "One",
    }

    with pytest.raises(judge.OpenAIJudgeError, match="differ"):
        judge._validate_and_score_decision(decision, judge_input)


def test_exact_excerpt_list_schema_requires_an_array() -> None:
    decision_model = judge.build_judge_decision_model(
        ["K1"],
        answer_evidence_format="exact_excerpt_list",
    )

    parsed = decision_model.model_validate(
        {
            "keypoint_evaluations": {
                "K1": {
                    "label": "covered",
                    "rationale": "Two locations support the composite keypoint.",
                    "answer_evidence": ["first excerpt", "second excerpt"],
                }
            },
            "unsupported_claims": [],
            "overall_assessment": "Covered.",
        }
    )

    assert parsed.keypoint_evaluations.K1.answer_evidence == [
        "first excerpt",
        "second excerpt",
    ]


def test_segment_id_schema_and_callable_resolve_exact_candidate_evidence() -> None:
    captured: dict = {}

    class FakeResponses:
        def parse(self, **kwargs):
            captured.update(kwargs)
            parsed = kwargs["text_format"].model_validate(
                {
                    "keypoint_evaluations": {
                        "K1": {
                            "label": "covered",
                            "rationale": "Both facts occur in candidate segments.",
                            "answer_evidence_segment_ids": ["A0001", "A0003"],
                        }
                    },
                    "unsupported_claims": [],
                    "overall_assessment": "Covered.",
                }
            )
            return SimpleNamespace(
                id="resp-segmented",
                model="gpt-test",
                output_parsed=parsed,
                output_text=parsed.model_dump_json(),
                usage=None,
            )

    api_judge = judge.OpenAIKeypointJudge(
        prompt={
            "system_prompt": "Evaluate numbered candidate segments.",
            "user_prompt_template": "{evaluation_item_json}",
            "judge_input_policy": "keypoint_segmented_reference_isolated",
            "judge_input_fields": [
                "qra_id",
                "language",
                "question",
                "answerable",
                "keypoints",
                "candidate_answer",
            ],
            "answer_evidence_format": "segment_id_list",
            "unsupported_claims_evaluation": "not_evaluated",
        },
        model="gpt-test",
        client=SimpleNamespace(responses=FakeResponses()),
    )

    result = api_judge(
        {
            "qra_id": "TEST-F-001",
            "language": "de",
            "question": "Welche Punkte gelten?",
            "answerable": True,
            "keypoints": [{"id": "K1", "text": "Erster und dritter Punkt"}],
            "candidate_answer": "Erster Punkt.\n\nZweiter Punkt.\n\nDritter Punkt.",
            "reference_answer": "Secret reference text.",
            "reference_evidence": [{"reference_id": "R1", "quote": "Secret"}],
        }
    )

    visible_input = captured["input"][1]["content"]
    assert "candidate_answer_segments" in visible_input
    assert '"candidate_answer"' not in visible_input
    assert "reference_answer" not in visible_input
    evaluation = result["decision"]["keypoint_evaluations"]["K1"]
    assert evaluation["answer_evidence_segment_ids"] == ["A0001", "A0003"]
    assert evaluation["answer_evidence"] == ["Erster Punkt.", "Dritter Punkt."]


def test_segment_id_schema_rejects_ids_absent_from_candidate() -> None:
    decision_model = judge.build_judge_decision_model(
        ["K1"],
        answer_evidence_format="segment_id_list",
        candidate_segment_ids=["A0001", "A0002"],
    )

    with pytest.raises(ValidationError):
        decision_model.model_validate(
            {
                "keypoint_evaluations": {
                    "K1": {
                        "label": "covered",
                        "rationale": "Invented shortened ID.",
                        "answer_evidence_segment_ids": ["A002"],
                    }
                },
                "unsupported_claims": [],
                "overall_assessment": "",
            }
        )


@pytest.mark.parametrize("label", ["covered", "contradicted"])
def test_decisive_label_rejects_evidence_copied_from_reference(label: str) -> None:
    """A Judge may not score reference text that is absent from the answer."""

    decision = {
        "keypoint_evaluations": [
            {
                "keypoint_id": "K1",
                "label": label,
                "rationale": "The deadline is supposedly present.",
                "answer_evidence": "innerhalb eines Monats nach Neugründung",
            }
        ],
        "unsupported_claims": [],
        "overall_assessment": "",
    }
    judge_input = {
        "keypoints": [
            {
                "id": "K1",
                "text": "Der steuerliche Erfassungsfragebogen folgt innerhalb eines Monats.",
            }
        ],
        "reference_answer": (
            "Der Fragebogen ist innerhalb eines Monats nach Neugründung zu übermitteln."
        ),
        "candidate_answer": (
            "Fülle nach der Gewerbeanmeldung den Fragebogen zur steuerlichen "
            "Erfassung aus."
        ),
    }

    with pytest.raises(
        judge.OpenAIJudgeError,
        match="K1.*exact candidate-answer excerpt",
    ):
        judge._validate_and_score_decision(decision, judge_input)


@pytest.mark.parametrize("label", ["covered", "contradicted"])
def test_decisive_label_requires_nonempty_answer_evidence(label: str) -> None:
    decision = {
        "keypoint_evaluations": [
            {
                "keypoint_id": "K1",
                "label": label,
                "rationale": "A decisive label needs auditable evidence.",
                "answer_evidence": "",
            }
        ],
        "unsupported_claims": [],
        "overall_assessment": "",
    }
    judge_input = {
        "keypoints": [{"id": "K1", "text": "Punkt eins"}],
        "candidate_answer": "Punkt eins ist enthalten.",
    }

    with pytest.raises(
        judge.OpenAIJudgeError,
        match="K1.*exact candidate-answer excerpt",
    ):
        judge._validate_and_score_decision(decision, judge_input)


def test_decisive_label_accepts_exact_candidate_evidence() -> None:
    decision = {
        "keypoint_evaluations": [
            {
                "keypoint_id": "K1",
                "label": "covered",
                "rationale": "The answer states the deadline.",
                "answer_evidence": "innerhalb eines Monats",
            }
        ],
        "unsupported_claims": [],
        "overall_assessment": "",
    }
    judge_input = {
        "keypoints": [{"id": "K1", "text": "Frist von einem Monat"}],
        "candidate_answer": "Der Fragebogen ist innerhalb eines Monats einzureichen.",
    }

    result = judge._validate_and_score_decision(decision, judge_input)

    assert result["metrics"]["completeness"] == 1.0
    assert result["validation_warnings"] == []


def test_decisive_label_accepts_multiple_exact_candidate_excerpts() -> None:
    decision = {
        "keypoint_evaluations": [
            {
                "keypoint_id": "K1",
                "label": "covered",
                "rationale": "Both required elements are explicit in separate places.",
                "answer_evidence": ["einen Modellkunden erstellen", "- Einkommen"],
            }
        ],
        "unsupported_claims": [],
        "overall_assessment": "",
    }
    judge_input = {
        "keypoints": [{"id": "K1", "text": "Modellkunde mit Einkommensmerkmal"}],
        "candidate_answer": (
            "Du solltest einen Modellkunden erstellen.\nMerkmale:\n- Einkommen"
        ),
    }

    result = judge._validate_and_score_decision(decision, judge_input)

    assert result["metrics"]["completeness"] == 1.0
    assert result["keypoint_evaluations"][0]["answer_evidence"] == [
        "einen Modellkunden erstellen",
        "- Einkommen",
    ]


def test_evidence_list_rejects_a_rewritten_excerpt() -> None:
    decision = {
        "keypoint_evaluations": [
            {
                "keypoint_id": "K1",
                "label": "covered",
                "rationale": "The wording was changed.",
                "answer_evidence": ["termingenaue Gegen眉berstellung"],
            }
        ],
        "unsupported_claims": [],
        "overall_assessment": "",
    }
    judge_input = {
        "keypoints": [{"id": "K1", "text": "Termingenaue Gegen眉berstellung"}],
        "candidate_answer": "Stelle Ein- und Auszahlungen termingenau gegen眉ber.",
    }

    with pytest.raises(
        judge.OpenAIJudgeError,
        match="K1 answer_evidence must contain only exact candidate-answer excerpts",
    ):
        judge._validate_and_score_decision(decision, judge_input)


def test_neither_label_rejects_non_candidate_answer_evidence() -> None:
    decision = {
        "keypoint_evaluations": [
            {
                "keypoint_id": "K1",
                "label": "neither",
                "rationale": "The keypoint is absent.",
                "answer_evidence": "Nur im Referenztext enthalten",
            }
        ],
        "unsupported_claims": [],
        "overall_assessment": "",
    }
    judge_input = {
        "keypoints": [{"id": "K1", "text": "Punkt zwei"}],
        "candidate_answer": "Punkt eins ist enthalten.",
    }

    with pytest.raises(
        judge.OpenAIJudgeError,
        match="K1 answer_evidence must be an exact candidate-answer excerpt",
    ):
        judge._validate_and_score_decision(decision, judge_input)


def test_neither_label_accepts_exact_partial_match_evidence() -> None:
    decision = {
        "keypoint_evaluations": [
            {
                "keypoint_id": "K1",
                "label": "neither",
                "rationale": "The topic is named but the deadline is absent.",
                "answer_evidence": "Fragebogen ausf眉llen",
            }
        ],
        "unsupported_claims": [],
        "overall_assessment": "",
    }
    judge_input = {
        "keypoints": [
            {"id": "K1", "text": "Fragebogen innerhalb eines Monats einreichen"}
        ],
        "candidate_answer": "Nach der Anmeldung den Fragebogen ausf眉llen.",
    }

    result = judge._validate_and_score_decision(decision, judge_input)

    assert result["metrics"]["irrelevancy"] == 1.0
    assert result["keypoint_evaluations"][0]["answer_evidence"] == (
        "Fragebogen ausf眉llen"
    )


def test_local_validation_failure_retains_api_audit_data() -> None:
    candidate = {
        "judge_input": {
            "keypoints": [{"id": "K1", "text": "One"}],
            "candidate_answer": "One",
        }
    }

    def invalid_api_result(_judge_input: dict) -> dict:
        return {
            "response_id": "resp-paid",
            "response_model": "gpt-test-snapshot",
            "raw_output_text": '{"keypoint_evaluations": {}}',
            "decision": {
                "keypoint_evaluations": {},
                "unsupported_claims": [],
                "overall_assessment": "",
            },
            "usage": {
                "input_tokens": 50,
                "output_tokens": 10,
                "total_tokens": 60,
            },
        }

    result = judge._safe_judge_call(
        candidate=candidate,
        judge_fn=invalid_api_result,
    )

    assert result["status"] == "error"
    assert result["response_id"] == "resp-paid"
    assert result["response_model"] == "gpt-test-snapshot"
    assert result["usage"]["total_tokens"] == 60
    assert result["raw_output_text"] == '{"keypoint_evaluations": {}}'
    assert judge._usage_totals([{"judge": result}])["total_tokens"] == 60


def test_run_exports_blinded_scores_comparison_and_manifest(
    tmp_path: Path,
) -> None:
    source_path = _source_run_fixture(tmp_path)
    prompt_path = _prompt_fixture(tmp_path)
    events: list[dict] = []

    result = judge.run_openai_judge(
        source_run_path=source_path,
        output_root=tmp_path / "judge-runs",
        tested_models=["test:model"],
        qra_ids=["TEST-F-001"],
        run_id="JUDGE-TEST-001",
        prompt_path=prompt_path,
        judge_fn=_decision_for,
        progress_callback=events.append,
    )

    report = result["report"]
    assert report["status"] == "completed"
    assert len(report["results"]) == 2
    assert report["usage"] == {
        "input_tokens": 200,
        "cached_input_tokens": 0,
        "output_tokens": 60,
        "reasoning_tokens": 20,
        "total_tokens": 260,
    }
    paired = report["paired_comparison"]["summary"][0]
    assert paired["paired_qra_count"] == 1
    assert paired["mean_rag_minus_baseline_completeness"] == pytest.approx(0.3334)
    assert paired["mean_rag_minus_baseline_hallucination"] == pytest.approx(-0.3333)
    assert [event["type"] for event in events] == [
        "start",
        "complete",
        "start",
        "complete",
    ]

    run_dir = Path(result["run_dir"])
    assert Path(result["scores_path"]).exists()
    assert Path(result["markdown_path"]).exists()
    assert Path(result["manifest_path"]).exists()
    assert not list(run_dir.glob("*.partial.json"))

    blinded_path = run_dir / "JUDGE-TEST-001_blinded_inputs.json"
    mapping_path = run_dir / "JUDGE-TEST-001_candidate_mapping.json"
    blinded = json.loads(blinded_path.read_text(encoding="utf-8"))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    assert "tested_model" not in json.dumps(blinded)
    assert mapping["candidates"][0]["tested_model"] == "test:model"

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"].values():
        artifact_path = run_dir / artifact["path"]
        assert artifact_path.exists()
        assert judge.sha256_file(artifact_path) == artifact["sha256"]


def test_run_records_non_openai_provider_without_api_key(tmp_path: Path) -> None:
    source_path = _source_run_fixture(tmp_path)
    prompt_path = _prompt_fixture(tmp_path)

    result = judge.run_openai_judge(
        source_run_path=source_path,
        output_root=tmp_path / "judge-runs",
        tested_models=["test:model"],
        qra_ids=["TEST-F-001"],
        modes=["baseline"],
        run_id="DEEPSEEK-JUDGE-TEST-001",
        prompt_path=prompt_path,
        judge_fn=_decision_for,
        judge_provider="deepseek",
        api_base_url="https://api.deepseek.example",
    )

    recorded = result["report"]["judge"]
    assert recorded["provider"] == "deepseek"
    assert recorded["base_url"] == "https://api.deepseek.example"
    assert "api_key" not in recorded


def test_resume_retries_only_failed_candidate(tmp_path: Path) -> None:
    source_path = _source_run_fixture(tmp_path)
    prompt_path = _prompt_fixture(tmp_path)
    output_root = tmp_path / "judge-runs"
    run_id = "JUDGE-RESUME-001"
    call_count = 0

    def fail_second_call(judge_input: dict) -> dict:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("simulated transient failure")
        return _decision_for(judge_input)

    with pytest.raises(judge.OpenAIJudgeError, match="simulated transient failure"):
        judge.run_openai_judge(
            source_run_path=source_path,
            output_root=output_root,
            tested_models=["test:model"],
            qra_ids=["TEST-F-001"],
            run_id=run_id,
            prompt_path=prompt_path,
            judge_fn=fail_second_call,
            fail_on_error=True,
        )

    partial_path = output_root / run_id / f"{run_id}.partial.json"
    partial = json.loads(partial_path.read_text(encoding="utf-8"))
    assert partial["status"] == "failed"
    assert [item["judge"]["status"] for item in partial["results"]] == [
        "success",
        "error",
    ]

    resumed_inputs: list[dict] = []

    def resumed_judge(judge_input: dict) -> dict:
        resumed_inputs.append(judge_input)
        return _decision_for(judge_input)

    result = judge.run_openai_judge(
        source_run_path=source_path,
        output_root=output_root,
        tested_models=["test:model"],
        qra_ids=["TEST-F-001"],
        run_id=run_id,
        prompt_path=prompt_path,
        judge_fn=resumed_judge,
        fail_on_error=True,
        resume=True,
    )

    assert len(resumed_inputs) == 1
    assert result["report"]["status"] == "completed"
    assert result["report"]["resume"] == {
        "resumed": True,
        "resume_count": 1,
        "preexisting_successful_answers": 1,
    }
    assert len(result["report"]["prior_attempts"]) == 1
    assert (
        result["report"]["prior_attempts"][0]["judge"]["error"]
        == "simulated transient failure"
    )
    assert len(result["report"]["results"]) == 2
    assert not partial_path.exists()
