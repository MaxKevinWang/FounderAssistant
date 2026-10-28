from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import FounderProfileViewSet, UserViewSet

router = DefaultRouter()
router.register(r"users", UserViewSet, basename="user")
router.register(r"profiles", FounderProfileViewSet, basename="founder-profile")

urlpatterns = [
    path("", include(router.urls)),
]
