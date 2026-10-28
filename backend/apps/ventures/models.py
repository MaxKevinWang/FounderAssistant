from django.conf import settings
from django.db import models


class Venture(models.Model):
    STAGE_CHOICES = [
        ("idea", "Idea"),
        ("validation", "Validation"),
        ("mvp", "MVP"),
        ("go_to_market", "Go-To-Market"),
        ("scale", "Scale"),
    ]

    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="ventures")
    name = models.CharField(max_length=200)
    stage = models.CharField(max_length=32, choices=STAGE_CHOICES, default="idea")
    problem_statement = models.TextField(blank=True)
    value_proposition = models.TextField(blank=True)
    target_segment = models.TextField(blank=True)
    legal_form = models.CharField(max_length=64, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self) -> str:
        return self.name


class VentureMilestone(models.Model):
    venture = models.ForeignKey(Venture, on_delete=models.CASCADE, related_name="milestones")
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    due_date = models.DateField(null=True, blank=True)
    is_completed = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["is_completed", "due_date"]

    def __str__(self) -> str:
        return f"{self.venture.name}: {self.title}"
