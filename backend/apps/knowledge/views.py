from rest_framework import viewsets
from rest_framework.permissions import IsAuthenticated

from .models import KnowledgeChunk, KnowledgeDocument
from .serializers import KnowledgeChunkSerializer, KnowledgeDocumentSerializer


class KnowledgeDocumentViewSet(viewsets.ModelViewSet):
    serializer_class = KnowledgeDocumentSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return KnowledgeDocument.objects.filter(owner=self.request.user).prefetch_related("chunks")

    def perform_create(self, serializer):
        serializer.save(owner=self.request.user)


class KnowledgeChunkViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = KnowledgeChunkSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return KnowledgeChunk.objects.filter(document__owner=self.request.user).select_related("document")
