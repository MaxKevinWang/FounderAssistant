from rest_framework import serializers

from .models import KnowledgeChunk, KnowledgeDocument


class KnowledgeChunkSerializer(serializers.ModelSerializer):
    class Meta:
        model = KnowledgeChunk
        fields = ["id", "chunk_index", "chunk_text", "metadata"]


class KnowledgeDocumentSerializer(serializers.ModelSerializer):
    chunks = KnowledgeChunkSerializer(many=True, read_only=True)

    class Meta:
        model = KnowledgeDocument
        fields = [
            "id",
            "venture",
            "owner",
            "source_type",
            "title",
            "source_uri",
            "language",
            "content",
            "embedding_status",
            "created_at",
            "chunks",
        ]
        read_only_fields = ["owner"]
