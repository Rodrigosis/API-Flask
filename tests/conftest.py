import pytest

from wine import create_app


@pytest.fixture(scope="session")
def app():
    """Create and configure a new app instance for each test."""
    app = create_app()

    with app.app_context():
        yield app


@pytest.fixture(scope="session")
def client(app):
    """A test client for the app."""
    return app.test_client()
