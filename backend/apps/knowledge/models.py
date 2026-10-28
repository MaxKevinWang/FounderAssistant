from django.conf import settings
from django.db import models

from apps.ventures.models import Venture


class KnowledgeDocument(models.Model):
    SOURCE_TYPES = [
        ("uploaded", "Uploaded"),
        ("url", "URL"),
        ("regulation", "Regulation"),
        ("template", "Template"),
    ]

    venture = models.ForeignKey(Venture, on_delete=models.CASCADE, related_name="knowledge_documents")
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="knowledge_documents")
    source_type = models.CharField(max_length=32, choices=SOURCE_TYPES, default="uploaded")
    title = models.CharField(max_length=255)
    source_uri = models.URLField(blank=True)
    language = models.CharField(max_length=8, default="de")
    content = models.TextField(blank=True)
    embedding_status = models.CharField(max_length=32, default="pending")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.title


class KnowledgeChunk(models.Model):
    document = models.ForeignKey(KnowledgeDocument, on_delete=models.CASCADE, related_name="chunks")
    chunk_index = models.PositiveIntegerField()
    chunk_text = models.TextField()
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        unique_together = ("document", "chunk_index")
        ordering = ["document_id", "chunk_index"]

    def __str__(self) -> str:
        return f"Chunk {self.chunk_index} of {self.document_id}"
