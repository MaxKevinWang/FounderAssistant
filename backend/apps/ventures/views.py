from rest_framework import viewsets
from rest_framework.exceptions import PermissionDenied
from rest_framework.permissions import IsAuthenticated

from .models import Venture, VentureMilestone
from .serializers import VentureMilestoneSerializer, VentureSerializer


class VentureViewSet(viewsets.ModelViewSet):
    serializer_class = VentureSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Venture.objects.filter(owner=self.request.user).prefetch_related("milestones")

    def perform_create(self, serializer):
        serializer.save(owner=self.request.user)


class VentureMilestoneViewSet(viewsets.ModelViewSet):
    serializer_class = VentureMilestoneSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return VentureMilestone.objects.filter(venture__owner=self.request.user).select_related("venture")

    def perform_create(self, serializer):
        venture = serializer.validated_data["venture"]
        if venture.owner_id != self.request.user.id:
            raise PermissionDenied("You can only create milestones for your own ventures.")
        serializer.save()
