import pytest

from apps.users.models import User
from apps.ventures.models import Venture


@pytest.mark.django_db
def test_create_venture():
    user = User.objects.create_user(username="alice", password="pwd12345")
    venture = Venture.objects.create(owner=user, name="Acme", stage="idea")

    assert venture.owner == user
    assert venture.name == "Acme"
