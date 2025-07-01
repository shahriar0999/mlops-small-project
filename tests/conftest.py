import pytest
import mlflow
import os
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

@pytest.fixture(scope="session")
def model_and_data():
    # setup dagshub credentials for mlflow tracking
    dagshub_token = os.getenv("DAGSHUB_PAT")
    if not dagshub_token:
        raise EnvironmentError("DAGSHUB_PAT environment variable is not set")
    
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = 'shahriar0999'
    repo_name = 'mlops-small-project'

    # Set up MLflow tracking URI
    mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

    # Load the latest model version from MLflow model registry
    model_name = "own_model"
    model_version = get_latest_model_version(model_name)
    if model_version is None:
        raise ValueError(f"No model found in stage 'Staging' for model '{model_name}'")
    
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)

    # load the vectorizer
    vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

    # load the test data
    test_data = pd.read_csv("data/features/test_bow.csv")

    return model, vectorizer, test_data


def get_latest_model_version(model_name, stage='Production'):
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=[stage])
    return latest_version[0].version if latest_version else None