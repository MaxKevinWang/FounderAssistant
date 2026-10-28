"""Django admin configuration for evaluation entities."""

from django.contrib import admin

from .models import ComparisonResult, EvaluationArtifact, EvaluationRun, ScenarioCase


@admin.register(ScenarioCase)
class ScenarioCaseAdmin(admin.ModelAdmin):
    """Admin list setup for scenario test cases."""

    list_display = ("id", "code", "domain", "title", "created_at")
    list_filter = ("domain", "created_at")
    search_fields = ("code", "title", "problem_statement")
    readonly_fields = ("created_at",)


@admin.register(EvaluationRun)
class EvaluationRunAdmin(admin.ModelAdmin):
    """Admin list setup for single baseline/RAG run records."""

    list_display = ("id", "run_name", "owner", "venture", "run_mode", "decision_domain", "created_at")
    list_filter = ("run_mode", "decision_domain", "created_at")
    search_fields = ("run_name", "owner__username", "venture__name", "llm_model")
    readonly_fields = ("created_at",)


@admin.register(EvaluationArtifact)
class EvaluationArtifactAdmin(admin.ModelAdmin):
    """Admin list setup for run artifacts (payload snapshots)."""

    list_display = ("id", "run", "artifact_type", "created_at")
    list_filter = ("artifact_type", "created_at")
    search_fields = ("artifact_type", "run__run_name")
    readonly_fields = ("created_at", "payload")


@admin.register(ComparisonResult)
class ComparisonResultAdmin(admin.ModelAdmin):
    """Admin list setup for baseline-vs-RAG comparison records."""

    list_display = ("id", "owner", "venture", "scenario", "winner", "created_at")
    list_filter = ("winner", "created_at")
    search_fields = ("owner__username", "venture__name", "scenario__code")
    readonly_fields = ("created_at",)
