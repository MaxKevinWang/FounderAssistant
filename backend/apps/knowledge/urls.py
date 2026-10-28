from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import KnowledgeChunkViewSet, KnowledgeDocumentViewSet

router = DefaultRouter()
router.register(r"documents", KnowledgeDocumentViewSet, basename="knowledge-document")
router.register(r"chunks", KnowledgeChunkViewSet, basename="knowledge-chunk")

urlpatterns = [path("", include(router.urls))]
