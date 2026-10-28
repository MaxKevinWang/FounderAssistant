from rest_framework import serializers

from .models import FounderProfile, User


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = [
            "id",
            "username",
            "email",
            "first_name",
            "last_name",
            "role",
            "preferred_language",
            "timezone",
        ]


class FounderProfileSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)

    class Meta:
        model = FounderProfile
        fields = [
            "id",
            "user",
            "venture_goal",
            "risk_tolerance",
            "industry_focus",
            "region",
        ]
