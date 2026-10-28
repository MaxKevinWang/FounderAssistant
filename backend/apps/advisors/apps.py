"""Django app registration for the startup advisor module.

`apps.advisors` is the integration point for:
1) conversation persistence models,
2) baseline local LLM chat endpoints,
3) local RAG build/retrieval endpoints used by the thesis demo.
"""

from django.apps import AppConfig


class AdvisorsConfig(AppConfig):
    """Runtime metadata for Django to load the advisors app."""

    # Use BigAutoField so newly created tables default to bigint primary keys.
    default_auto_field = "django.db.models.BigAutoField"

    # Full dotted import path for this Django application.
    name = "apps.advisors"

    # Short internal app label used in migrations and admin registration.
    label = "advisors"
