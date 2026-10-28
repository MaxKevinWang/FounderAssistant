"""DRF serializers for scenario/evaluation/comparison API payloads."""

from rest_framework import serializers

from .models import ComparisonResult, EvaluationArtifact, EvaluationRun, ScenarioCase


class ScenarioCaseSerializer(serializers.ModelSerializer):
    """Serialize scenario metadata and expected-outcome structure."""

    class Meta:
        model = ScenarioCase
        fields = [
            "id",
            "code",
            "domain",
            "title",
            "problem_statement",
            "founder_context",
            "expected_outcomes",
            "created_at",
        ]


class EvaluationArtifactSerializer(serializers.ModelSerializer):
    """Serialize one artifact record linked to an evaluation run."""

    class Meta:
        model = EvaluationArtifact
        fields = ["id", "run", "artifact_type", "payload", "created_at"]


class EvaluationRunSerializer(serializers.ModelSerializer):
    """Serialize run-level metrics with nested scenario and artifacts."""

    artifacts = EvaluationArtifactSerializer(many=True, read_only=True)
    scenario_detail = ScenarioCaseSerializer(source="scenario", read_only=True)

    class Meta:
        model = EvaluationRun
        fields = [
            "id",
            "run_name",
            "owner",
            "venture",
            "scenario",
            "scenario_detail",
            "decision_domain",
            "run_mode",
            "llm_model",
            "retrieval_recall",
            "answer_faithfulness",
            "citation_precision",
            "relevance_score",
            "traceability_score",
            "perceived_usefulness",
            "response_latency_ms",
            "safety_flag",
            "notes",
            "created_at",
            "artifacts",
        ]
        read_only_fields = ["owner"]


class ComparisonResultSerializer(serializers.ModelSerializer):
    """Serialize baseline-vs-RAG comparison including run details."""

    scenario_detail = ScenarioCaseSerializer(source="scenario", read_only=True)
    baseline_run_detail = EvaluationRunSerializer(source="baseline_run", read_only=True)
    rag_run_detail = EvaluationRunSerializer(source="rag_run", read_only=True)

    class Meta:
        model = ComparisonResult
        fields = [
            "id",
            "owner",
            "venture",
            "scenario",
            "scenario_detail",
            "baseline_run",
            "baseline_run_detail",
            "rag_run",
            "rag_run_detail",
            "winner",
            "judge_notes",
            "created_at",
        ]
        read_only_fields = ["owner"]
