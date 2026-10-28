"""Django app registration for evaluation and benchmarking features."""

from django.apps import AppConfig


class EvaluationsConfig(AppConfig):
    """Runtime metadata for loading the evaluations app."""

    # Use bigint primary keys by default for new tables.
    default_auto_field = "django.db.models.BigAutoField"

    # Full import path of this Django app package.
    name = "apps.evaluations"

    # Internal short label used by Django ORM/admin/migrations.
    label = "evaluations"
