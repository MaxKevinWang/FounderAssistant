"""DRF serializers for advisor sessions/messages and nested API responses."""

from rest_framework import serializers

from .models import AdvisoryMessage, AdvisorySession


class AdvisoryMessageSerializer(serializers.ModelSerializer):
    """Serialize one persisted message record.

    Notes:
    - `citation_payload` is returned so the frontend can render evidence cards.
    - `provider/model_name` help distinguish baseline and RAG responses in logs/UI.
    """

    class Meta:
        model = AdvisoryMessage
        fields = [
            "id",
            "session",
            "role",
            "content",
            "provider",
            "model_name",
            "citation_payload",
            "created_at",
        ]


class AdvisorySessionSerializer(serializers.ModelSerializer):
    """Serialize a session with read-only nested message history."""

    # Nested read-only history to simplify client rendering in one request.
    messages = AdvisoryMessageSerializer(many=True, read_only=True)

    class Meta:
        model = AdvisorySession
        fields = [
            "id",
            "venture",
            "owner",
            "title",
            "active_module",
            "created_at",
            "messages",
        ]

        # Owner is always derived from authenticated request user in the viewset.
        read_only_fields = ["owner"]
