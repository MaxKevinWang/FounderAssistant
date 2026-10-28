from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import VentureMilestoneViewSet, VentureViewSet

router = DefaultRouter()
router.register(r"", VentureViewSet, basename="venture")
router.register(r"milestones", VentureMilestoneViewSet, basename="venture-milestone")

urlpatterns = [
    path("", include(router.urls)),
]
