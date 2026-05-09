# Gradient Boost Classifier
import numpy as np
import pandas as pd
import mlflow.sklearn
import dagshub
dagshub.init(repo_owner='email4prasanth', repo_name='end-to-end-mlops', mlflow=True)
mlflow.set_experiment("Experiment-3-mean")
# mlflow.set_tracking_uri("https://dagshub.com/email4prasanth/mlflow_exp_dagshub.mlflow") # is already handle by mlflow=True
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

# data collection no need to store
data = pd.read_csv(r"H:\MLOPS\ml_project_mlflow_pro\data\water_potability.csv")
from sklearn.model_selection import train_test_split
n_estimators = 100
test_size = 0.2
train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)

# data prepration fill the missing values using mean values
def fill_missing_with_mean(df):
    for column in df.columns:
        if df[column].isnull().any():
            mean_value = df[column].mean()
            df[column].fillna(mean_value,inplace=True)
    return df
train_processed_data = fill_missing_with_mean(train_data)
test_processed_data = fill_missing_with_mean(test_data)

# split the data in training and testing
X_train = train_processed_data.drop(columns = ['Potability'], axis=1)
y_train = train_processed_data['Potability']
X_test = test_processed_data.drop(columns = ['Potability'], axis=1)
y_test = test_processed_data['Potability']

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier

# Models Dictionary
models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(),
    "random_forest": RandomForestClassifier(),
    "svm": SVC(),
    "knn": KNeighborsClassifier(),
    "naive_bayes": GaussianNB(),
    "gradientboost": GradientBoostingClassifier(),
}

with mlflow.start_run(run_name="water potability models experiment"):
    # Iterative over each model
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name, nested=True):
            # Train the model on training data
            model.fit(X_train, y_train)

            # save trained model using pickel
            model_filename = f"{model_name.replace(' ','-')}.pkl"
            pickle.dump(model,open("model.pkl","wb"))

            # Make predictions on test data
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
            plt.title(f"Confusion matrix for {model_name}")
            plt.savefig(f"Confusion_matrix_{model_name.replace(' ','-')}.png")
            plt.close()
            mlflow.log_artifact(f"Confusion_matrix_{model_name.replace(' ','-')}.png")

            mlflow.sklearn.log_model(model, model_name.replace(' ','-') )
            mlflow.log_artifact(__file__)

            mlflow.set_tag("owner","prasanth")
            mlflow.set_tag("env","all")

            print(f"Accuracy: {acc}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1-Score: {f1s}")
    
    print( "All models executed")
