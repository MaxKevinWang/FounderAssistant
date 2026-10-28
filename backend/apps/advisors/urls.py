"""URL routing for demo-focused advisor endpoints (LLM + RAG only)."""

from django.urls import path

from .views import (
    AdvisorRequestAssessmentView,
    LocalLLMChatView,
    LocalLLMStatusView,
    RAGBuildIndexView,
    RAGChunkListView,
    RAGChunkMetadataUpdateView,
    RAGChatView,
    RAGChatStreamView,
    RAGStatusView,
)

urlpatterns = [
    # Baseline local model endpoints (no retrieval augmentation).
    path("local-llm/status/", LocalLLMStatusView.as_view(), name="local-llm-status"),
    path("local-llm/chat/", LocalLLMChatView.as_view(), name="local-llm-chat"),
    path("request-assessment/", AdvisorRequestAssessmentView.as_view(), name="request-assessment"),

    # RAG lifecycle endpoints: readiness check, index build, grounded answer.
    path("rag/status/", RAGStatusView.as_view(), name="rag-status"),
    path("rag/build-index/", RAGBuildIndexView.as_view(), name="rag-build-index"),
    path("rag/chunks/", RAGChunkListView.as_view(), name="rag-chunk-list"),
    path("rag/chunks/metadata/", RAGChunkMetadataUpdateView.as_view(), name="rag-chunk-metadata-update"),
    path("rag/chat/", RAGChatView.as_view(), name="rag-chat"),
    path("rag/chat/stream/", RAGChatStreamView.as_view(), name="rag-chat-stream"),
]
