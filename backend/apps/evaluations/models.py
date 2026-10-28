"""Data models for benchmark scenarios, run records, and comparison outputs."""

from django.conf import settings
from django.db import models

from apps.ventures.models import Venture


class ScenarioCase(models.Model):
    """Canonical scenario definition used by evaluation and comparison flows."""

    DOMAIN_CHOICES = [
        ("legal_form", "Legal Form Selection"),
        ("funding", "Funding Preparation"),
        ("market_validation", "Market Validation"),
    ]

    code = models.CharField(max_length=64, unique=True)
    domain = models.CharField(max_length=32, choices=DOMAIN_CHOICES)
    title = models.CharField(max_length=255)
    problem_statement = models.TextField()
    founder_context = models.JSONField(default=dict, blank=True)
    expected_outcomes = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["domain", "code"]

    def __str__(self) -> str:
        return f"{self.code}: {self.title}"


class EvaluationRun(models.Model):
    """One executed evaluation run in either baseline or RAG mode."""

    MODE_CHOICES = [("baseline", "Baseline"), ("rag", "RAG")]
    DOMAIN_CHOICES = ScenarioCase.DOMAIN_CHOICES

    run_name = models.CharField(max_length=255)
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="evaluation_runs")
    venture = models.ForeignKey(Venture, on_delete=models.CASCADE, related_name="evaluation_runs")
    scenario = models.ForeignKey(
        ScenarioCase, on_delete=models.SET_NULL, null=True, blank=True, related_name="evaluation_runs"
    )
    decision_domain = models.CharField(max_length=32, choices=DOMAIN_CHOICES, default="legal_form")
    run_mode = models.CharField(max_length=16, choices=MODE_CHOICES, default="rag")
    llm_model = models.CharField(max_length=128)
    retrieval_recall = models.FloatField(default=0)
    answer_faithfulness = models.FloatField(default=0)
    citation_precision = models.FloatField(default=0)
    relevance_score = models.FloatField(default=0)
    traceability_score = models.FloatField(default=0)
    perceived_usefulness = models.FloatField(default=0)
    response_latency_ms = models.PositiveIntegerField(default=0)
    safety_flag = models.BooleanField(default=False)
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.run_name


class ComparisonResult(models.Model):
    """Pairwise comparison between one baseline run and one RAG run."""

    WINNER_CHOICES = [("baseline", "Baseline"), ("rag", "RAG"), ("tie", "Tie")]

    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="comparison_results")
    venture = models.ForeignKey(Venture, on_delete=models.CASCADE, related_name="comparison_results")
    scenario = models.ForeignKey(ScenarioCase, on_delete=models.CASCADE, related_name="comparison_results")
    baseline_run = models.ForeignKey(
        EvaluationRun, on_delete=models.CASCADE, related_name="baseline_comparison_results"
    )
    rag_run = models.ForeignKey(EvaluationRun, on_delete=models.CASCADE, related_name="rag_comparison_results")
    winner = models.CharField(max_length=16, choices=WINNER_CHOICES, default="rag")
    judge_notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"Comparison<{self.venture_id}:{self.scenario.code}>"


class EvaluationArtifact(models.Model):
    """Structured JSON artifact attached to one evaluation run."""

    run = models.ForeignKey(EvaluationRun, on_delete=models.CASCADE, related_name="artifacts")
    artifact_type = models.CharField(max_length=64)
    payload = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self) -> str:
        return f"Artifact<{self.artifact_type}>"
