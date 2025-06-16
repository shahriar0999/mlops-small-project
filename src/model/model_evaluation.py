import numpy as np
import pandas as pd
import logging
import pickle
import json
import os
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import mlflow, dagshub

# Load environment variables from .env file
load_dotenv()

# Set up DagsHub credentials for MLflow tracking
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

# Logging configuration
try:
    logger = logging.getLogger('model_evaluation')
    logger.setLevel('DEBUG')

    console_handler = logging.StreamHandler()
    console_handler.setLevel('DEBUG')

    file_handler = logging.FileHandler('model_evaluation.log')
    file_handler.setLevel('ERROR')

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
except Exception as e:
    print(f"Error configuring logging: {str(e)}")
    raise

def load_model():
    try:
        logger.info("Loading trained model")
        with open('models/model.pkl', 'rb') as f:
            model = pickle.load(f)
        logger.debug("Model loaded successfully")
        return model
    except FileNotFoundError:
        logger.error("Model file not found")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def load_test_data(data_path: str) -> pd.DataFrame:
    try:
        logger.info("Loading test data")
        test_data = pd.read_csv(data_path) # './data/features/test_bow.csv'
        
        X_test = test_data.iloc[:,0:-1].values
        y_test = test_data.iloc[:,-1].values
        
        logger.debug(f"Test data loaded successfully. Shape: {X_test.shape}")
        return X_test, y_test
    except FileNotFoundError:
        logger.error("Test data file not found")
        raise
    except Exception as e:
        logger.error(f"Error loading test data: {str(e)}")
        raise


def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logger.debug('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise


def main():
    mlflow.set_experiment("dvc-pipeline")
    with mlflow.start_run() as run:  # Start an MLflow run
        try:
            clf = load_model()
            X_test, y_test = load_test_data('./data/features/test_bow.csv')

            metrics = evaluate_model(clf, X_test, y_test)
            
            save_metrics(metrics, 'reports/metrics.json')
            
            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model parameters to MLflow
            if hasattr(clf, 'get_params'):
                params = clf.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)
            
            mlflow.log_artifact('models/model.pkl', artifact_path="models")

            # mlflow.sklearn.log_model(clf, "model")
            
            # Save model info
            save_model_info(run.info.run_id, "models", 'reports/model_info.json')
            
            # Log the metrics file to MLflow
            mlflow.log_artifact('reports/metrics.json')

            # Log the model info file to MLflow
            mlflow.log_artifact('reports/model_info.json')

            # Log the evaluation errors log file to MLflow
            # mlflow.log_artifact('model_evaluation_errors.log')
        except Exception as e:
            logger.error('Failed to complete the model evaluation process: %s', e)
            print(f"Error: {e}")

if __name__ == '__main__':
    main()