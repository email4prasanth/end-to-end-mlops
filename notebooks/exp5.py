import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

import mlflow
import dagshub
dagshub.init(repo_owner='email4prasanth', repo_name='end-to-end-mlops', mlflow=True)
mlflow.set_experiment("Experiment-5-hp-all_comb")

# data collection no need to store
data = pd.read_csv(r"H:\MLOPS\ml_project_jupiter\water_potability.csv")
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# data prepration using mean values
def fill_missing_with_median(df):
    for column in df.columns:
        if df[column].isnull().any():
            median_value = df[column].median()
            df[column].fillna(median_value,inplace=True)
    return df
# Fill the missing values with mean
train_processed_data = fill_missing_with_median(train_data)
test_processed_data = fill_missing_with_median(test_data)

# Prepare training data 
X_train = train_processed_data.drop(columns=['Potability'],axis=1)
y_train = train_processed_data['Potability']

# Define model and parameter distribution for Randomized Search CV
clf = RandomForestClassifier(random_state=42)
param_dist = {
    'n_estimators':[100, 200, 300, 500, 1000],
    'max_depth': [None, 20, 30, 40, 50],
}
# Perform Randomized Search CV to find best hyperparameter
random_search = RandomizedSearchCV(estimator=clf, param_distributions=param_dist, cv=5, n_jobs=-1, verbose=2, random_state=42)
with mlflow.start_run(run_name="RF Tunning") as parent_run:
    random_search.fit(X_train,y_train)
    for i in range(len(random_search.cv_results_['params'])):
        with mlflow.start_run(run_name=f"Combination{i+1}", nested=True) as child_run:
            mlflow.log_params(random_search.cv_results_['params'][i])
            mlflow.log_metric("mean_test_score",random_search.cv_results_['mean_test_score'][i])

    #Print the best hyperparameter found by RandomizedSearchCV
    print(f"Best parameters found: {random_search.best_params_}")
    mlflow.log_params(random_search.best_params_)

    # Train the model with best parameters
    best_rf = random_search.best_estimator_  
    best_rf.fit(X_train, y_train)

    # save 
    pickle.dump(best_rf,open("model.pkl","wb"))

    # Test model
    X_test = test_processed_data.drop(columns=['Potability'],axis=1)
    y_test = test_processed_data['Potability']

    # Load the saved model
    model = pickle.load(open('model.pkl',"rb"))
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1_s = f1_score(y_test,y_pred)

    #log metrics
    mlflow.log_metric("accuracy",acc)
    mlflow.log_metric("precision",precision)
    mlflow.log_metric("recall",recall)
    mlflow.log_metric("f1_score",f1_s)

    # log dataframe
    train_df = mlflow.data.from_pandas(train_processed_data)
    test_df = mlflow.data.from_pandas(test_processed_data)
    mlflow.log_input(train_df,"train")
    mlflow.log_input(test_df,"test")

    # log source code
    mlflow.log_artifact(__file__)
    # log model
    mlflow.sklearn.log_model(random_search.best_estimator_, "best_model")

    print(f"Accuracy: {acc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1_s}")

