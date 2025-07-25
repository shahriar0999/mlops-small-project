# 🚀 Sentiment Classification Pipeline with CI/CD and MLOps

This project demonstrates a complete end-to-end machine learning pipeline for sentiment analysis using modern MLOps practices. It includes data versioning (via DVC + S3), experiment tracking, API development and testing, containerization, deployment to AWS infrastructure, and automation via CI/CD.

---

## 📦 Features

- 🗃️ **Data Versioning & Storage:** Managed using [DVC](https://dvc.org/) with **Amazon S3** as the remote
- 🔬 **Experiment Tracking:** Integrated with [DagsHub](https://dagshub.com/) (via MLflow)
- 🧪 **Testing:** API testing using Pytest
- 🌐 **API Development:** RESTful API using Flask with simple HTML frontend
- 🐳 **Containerization:** Dockerized Flask app and pushed to AWS ECR
- ☁️ **Deployment:** Automatically deployed to EC2 (ASG) via AWS CodeDeploy
- ⚙️ **CI/CD:** Automated using GitHub Actions

---

## 🧱 Project Structure

```
.
├── apps/                     # Flask application code
├── notebooks/                # Experiments and analysis notebooks
├── src/                      # ML pipeline and modules
├── models/                   # Saved model and vectorizer
├── scripts/                  # Utility scripts (e.g., promote_model.py)
├── deploy/scripts/           # AWS deployment scripts (S3 download, etc.)
├── docs/                     # Documentation with Sphinx
├── tests/                    # Unit and integration tests
├── .dvc/                     # DVC pipeline and metadata
├── .github/workflows/        # GitHub Actions CI/CD workflow
├── Dockerfile                # Docker configuration for app
├── app.py                    # Main Flask API app
├── params.yaml               # ML pipeline parameters
├── dvc.yaml                  # DVC pipeline definition
├── requirements.txt          # Project dependencies
├── setup.py                  # Installable Python package
└── README.md                 # Project overview
```

---

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/your-repo.git
cd your-repo

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Set up DVC and pull data/models from S3
dvc pull
```

---

## 🚀 Running the App Locally

```bash
# Start the Flask app
python app.py
```

Visit: `http://localhost:8501`

---

## 🧪 Run Tests

```bash
pytest tests/
```

---

## 🧬 DVC Pipeline

```bash
# Run ML pipeline
dvc repro
```

All data and model artifacts are versioned and pulled from **Amazon S3** using DVC.

Check experiment tracking on [DagsHub](https://dagshub.com/shahriar0999/mlops-small-project.mlflow/#/experiments/0?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D).

---

## 📦 Docker

```bash
# Build Docker image
docker build -t sentiment-app .

# Run Docker container
docker run -p 8501:8501 sentiment-app
```

---

## ☁️ AWS Deployment Overview

1. Docker image is pushed to **AWS ECR**
2. **CodeDeploy** pulls deployment script from **S3**
3. Launches app into **EC2 (ASG)** setup
4. **CI/CD** automated using GitHub Actions

---

## 🔄 CI/CD (GitHub Actions)

- Triggered on push to `main`
- Includes:
  - Testing
  - Building Docker image
  - Pushing to ECR
  - Triggering deployment via CodeDeploy

Workflow file: `.github/workflows/ci.yaml`


---

## 🙏 Acknowledgments

- [DVC](https://dvc.org/)
- [Amazon S3](https://aws.amazon.com/s3/)
- [MLflow](https://mlflow.org/)
- [DagsHub](https://dagshub.com/)
- [AWS CodeDeploy](https://aws.amazon.com/codedeploy/)
- [Flask](https://flask.palletsprojects.com/)

---

## 📬 Contact

For questions or collaboration, please reach out via GitHub Issues or open a pull request!
