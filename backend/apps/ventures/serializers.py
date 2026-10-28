from rest_framework import serializers

from .models import Venture, VentureMilestone


class VentureMilestoneSerializer(serializers.ModelSerializer):
    class Meta:
        model = VentureMilestone
        fields = [
            "id",
            "venture",
            "title",
            "description",
            "due_date",
            "is_completed",
            "created_at",
        ]


class VentureSerializer(serializers.ModelSerializer):
    milestones = VentureMilestoneSerializer(many=True, read_only=True)

    class Meta:
        model = Venture
        fields = [
            "id",
            "owner",
            "name",
            "stage",
            "problem_statement",
            "value_proposition",
            "target_segment",
            "legal_form",
            "created_at",
            "updated_at",
            "milestones",
        ]
        read_only_fields = ["owner"]
