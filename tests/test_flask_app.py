import pytest 
import sys
import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# setup dagshub credentials for mlflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")
    

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from apps.app import app as flask_app


@pytest.fixture(scope="module")
def client():
    with flask_app.test_client() as client:
        yield client

def test_home_page_status(client):
    response = client.get('/')
    assert response.status_code == 200, "Home page should return status code 200"
    assert b'<title>Sentiment Analysis</title>' in response.data


def test_predict_page(client):
    response = client.post('/predict', data=dict(text="I love this!"))
    assert response.status_code == 200
    assert b'Happy' in response.data or b'Sad' in response.data, \
        "Response should contain either 'Happy' or 'Sad'"


if __name__ == '__main__':
    pytest.main()