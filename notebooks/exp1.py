# Gradient Boost Classifier
import numpy as np
import pandas as pd

# data collection no need to store
data = pd.read_csv(r"H:\MLOPS\ml_project_mlflow_pro\data\water_potability.csv")
from sklearn.model_selection import train_test_split
n_estimators = 100
test_size = 0.2
train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)

# data prepration using mean values
def fill_missing_with_mean(df):
    for column in df.columns:
        if df[column].isnull().any():
            mean_value = df[column].mean()
            df[column].fillna(mean_value,inplace=True)
    return df
train_processed_data = fill_missing_with_mean(train_data)
test_processed_data = fill_missing_with_mean(test_data)

import mlflow.sklearn
import dagshub
dagshub.init(repo_owner='email4prasanth', repo_name='end-to-end-mlops', mlflow=True)
mlflow.set_experiment("Experiment-1")
# mlflow.set_tracking_uri("https://dagshub.com/email4prasanth/mlflow_exp_dagshub.mlflow") # is already handle by mlflow=True
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

with mlflow.start_run():
    # Create a model 
    from sklearn.ensemble import GradientBoostingClassifier
    import pickle
    X_train = train_processed_data.drop(columns = ['Potability'], axis=1)
    y_train = train_processed_data['Potability']
    clf = GradientBoostingClassifier(n_estimators=n_estimators)
    clf.fit(X_train,y_train)

    mlflow.sklearn.log_model(clf, artifact_path="model")

    # save 
    pickle.dump(clf,open("model.pkl","wb"))
    mlflow.log_artifact("model.pkl")

    # Test model
    X_test = test_processed_data.drop(columns = ['Potability'], axis=1)
    y_test = test_processed_data['Potability']

    train_df = mlflow.data.from_pandas(train_processed_data)
    test_df = mlflow.data.from_pandas(test_processed_data)

    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
    model = pickle.load(open('model.pkl',"rb"))
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1s = f1_score(y_test,y_pred)

    mlflow.log_metric("Accuracy", acc)
    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("Recall", recall)
    mlflow.log_metric("F1-Score", f1s)

    mlflow.log_param("n_estimators",n_estimators)
    mlflow.log_param("test_size", test_size)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True)
    plt.xlabel("Prediction")
    plt.ylabel("Actual")
    plt.title("Confusion matrix")
    plt.savefig("Confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("Confusion_matrix.png")
    
    mlflow.sklearn.log_model(clf,"GradientBoostingClassifier")
    mlflow.log_artifact(__file__)
    mlflow.log_input(train_df, "train")
    mlflow.log_input(test_df, "test")

    mlflow.set_tag("owner","prasanth")
    mlflow.set_tag("env","all")

    print(f"Accuracy: {acc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1s}")
