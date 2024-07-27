import dagshub
dagshub.init(repo_owner='kalehariprasad', repo_name='mlflow-dagshub-demo', mlflow=True)

import mlflow
mlflow.set_tracking_uri('https://dagshub.com/kalehariprasad/mlflow-dagshub-demo.mlflow')
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)
  