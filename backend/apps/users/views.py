from rest_framework import mixins, viewsets
from rest_framework.permissions import IsAuthenticated

from .models import FounderProfile, User
from .serializers import FounderProfileSerializer, UserSerializer


class UserViewSet(mixins.ListModelMixin, mixins.RetrieveModelMixin, viewsets.GenericViewSet):
    serializer_class = UserSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return User.objects.filter(id=self.request.user.id)


class FounderProfileViewSet(viewsets.ModelViewSet):
    serializer_class = FounderProfileSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return FounderProfile.objects.filter(user=self.request.user).select_related("user")

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)
