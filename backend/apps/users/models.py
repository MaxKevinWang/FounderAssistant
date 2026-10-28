from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    role = models.CharField(max_length=64, default="founder")
    preferred_language = models.CharField(max_length=8, default="de")
    timezone = models.CharField(max_length=64, default="Europe/Berlin")


class FounderProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="founder_profile")
    venture_goal = models.CharField(max_length=255, blank=True)
    risk_tolerance = models.CharField(max_length=32, default="medium")
    industry_focus = models.CharField(max_length=128, blank=True)
    region = models.CharField(max_length=128, default="Germany")

    def __str__(self) -> str:
        return f"FounderProfile<{self.user.username}>"
