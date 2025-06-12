import mlflow
import dagshub 

# set tracking uri
mlflow.set_tracking_uri("https://dagshub.com/shahriar0999/mlops-small-project.mlflow")
dagshub.init(repo_owner='shahriar0999', repo_name='mlops-small-project', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)