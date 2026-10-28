"""Django admin configuration for advisor conversation models.

Admin is mainly used during demo/debug to:
1) inspect whether messages are being persisted correctly,
2) verify owner/session relationships,
3) inspect RAG citation payloads returned by the backend.
"""

from django.contrib import admin

from .models import AdvisoryMessage, AdvisorySession


@admin.register(AdvisorySession)
class AdvisorySessionAdmin(admin.ModelAdmin):
    """List configuration for conversation sessions."""

    # Show core ownership and routing information at a glance.
    list_display = ("id", "title", "owner", "venture", "active_module", "created_at")

    # Common filters for quickly narrowing sessions by module and creation window.
    list_filter = ("active_module", "created_at")

    # Enables searching session title, owner username, and venture title.
    search_fields = ("title", "owner__username", "venture__name")

    # Creation time is system-generated and should not be editable from admin.
    readonly_fields = ("created_at",)


@admin.register(AdvisoryMessage)
class AdvisoryMessageAdmin(admin.ModelAdmin):
    """List configuration for individual chat messages."""

    # Keep table concise while still showing the provider/model that generated output.
    list_display = ("id", "session", "role", "provider", "model_name", "created_at")

    # Filter by role and model to inspect baseline vs RAG generated records.
    list_filter = ("role", "provider", "model_name", "created_at")

    # Text search over content helps debugging prompt/response behavior.
    search_fields = ("content", "session__title", "session__owner__username")

    # Citation payload and timestamps should be read-only in admin.
    readonly_fields = ("created_at", "citation_payload")
