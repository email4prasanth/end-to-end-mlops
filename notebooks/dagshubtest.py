import dagshub
import mlflow

mlflow.set_tracking_uri("https://dagshub.com/email4prasanth/end-to-end-mlops.mlflow")

dagshub.init(repo_owner='email4prasanth', repo_name='end-to-end-mlops', mlflow=True)

with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)
