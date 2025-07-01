import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# setup dagshub credentials for mlflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")
    


def test_model_loaded_properly(model_and_data):
    model, _, _ = model_and_data
    assert model is not None, "Model should not be None after loading."


def test_model_signature(model_and_data):
    model, vectorizer, _ = model_and_data

    # Create dummy input matching expected vectorizer features
    input_text = "hi how are you"
    input_data = vectorizer.transform([input_text])
    input_df = pd.DataFrame(input_data.toarray(), columns=[str(i) for i in range(input_data.shape[1])])

    prediction = model.predict(input_df)

    # Assert input shape matches vectorizer feature size
    assert input_df.shape[1] == len(vectorizer.get_feature_names_out()), \
        "Input shape does not match expected number of features from vectorizer."

    # Assert output shape
    assert len(prediction) == input_df.shape[0], \
        f"Prediction length {len(prediction)} does not match input rows {input_df.shape[0]}."
    assert prediction.ndim == 1, "Prediction should be a 1D array for binary classification."



def test_model_performance(model_and_data):
    model, _, holdout_data = model_and_data

    X_test = holdout_data.iloc[:, 0:-1]
    y_test = holdout_data.iloc[:, -1]

    y_pred_new = model.predict(X_test)

    # Compute metrics
    accuracy_new = accuracy_score(y_test, y_pred_new)
    precision_new = precision_score(y_test, y_pred_new)
    recall_new = recall_score(y_test, y_pred_new)
    f1_new = f1_score(y_test, y_pred_new)

    # Define expected thresholds
    expected_accuracy = 0.40
    expected_precision = 0.40
    expected_recall = 0.40
    expected_f1 = 0.40

    # Assertions
    assert accuracy_new >= expected_accuracy, f"Accuracy {accuracy_new} below threshold {expected_accuracy}"
    assert precision_new >= expected_precision, f"Precision {precision_new} below threshold {expected_precision}"
    assert recall_new >= expected_recall, f"Recall {recall_new} below threshold {expected_recall}"
    assert f1_new >= expected_f1, f"F1 {f1_new} below threshold {expected_f1}"