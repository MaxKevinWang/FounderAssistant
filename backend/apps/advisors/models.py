"""Data models for interactive advisor conversations.

The design separates session-level metadata from message-level content:
- AdvisorySession groups one conversational thread under one venture context.
- AdvisoryMessage stores each exchanged message plus optional citation payload.
"""

from django.conf import settings
from django.db import models

from apps.ventures.models import Venture


class AdvisorySession(models.Model):
    """Conversation container for one venture and one owner.

    `active_module` is used to tag the currently active advisory flow
    (for example: legal_form, funding, market_validation, general).
    """

    venture = models.ForeignKey(Venture, on_delete=models.CASCADE, related_name="advisory_sessions")
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="advisory_sessions")
    title = models.CharField(max_length=255)
    active_module = models.CharField(max_length=64, default="general")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.title


class AdvisoryMessage(models.Model):
    """Single chat message inside an advisory session.

    `citation_payload` is JSON to keep source references from RAG retrieval.
    For baseline responses this can stay empty.
    """

    ROLE_CHOICES = [("system", "System"), ("user", "User"), ("assistant", "Assistant")]

    session = models.ForeignKey(AdvisorySession, on_delete=models.CASCADE, related_name="messages")
    role = models.CharField(max_length=16, choices=ROLE_CHOICES)
    content = models.TextField()
    provider = models.CharField(max_length=64, blank=True)
    model_name = models.CharField(max_length=128, blank=True)
    citation_payload = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self) -> str:
        return f"{self.role} @ {self.created_at:%Y-%m-%d %H:%M:%S}"
