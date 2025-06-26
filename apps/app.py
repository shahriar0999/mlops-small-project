from flask import Flask, render_template, request
from preprocessing_utility import normalize_text
import os
import pickle
import dagshub
import mlflow
import dagshub
from dotenv import load_dotenv

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


app = Flask(__name__)

# load model from model registry
model_name = "own_model"
model_version = 5

model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.pyfunc.load_model(model_uri)

vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html", result=None)

@app.route("/predict", methods=['POST'])
def predict():
    text = request.form["text"]

    # clean
    text = normalize_text(text)

    # bow
    features = vectorizer.transform([text])

    # prediction
    result = model.predict(features)
    
    # result
    return render_template('index.html', result=result[0])



app.run(debug=True)