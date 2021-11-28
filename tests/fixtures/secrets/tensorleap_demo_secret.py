import os

import pytest


@pytest.fixture
def put_mock_secret_in_env():
    os.environ["SECRET"] = "secret mock"
    yield
    os.environ.pop("SECRET")
