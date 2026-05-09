# model evaluation
import pandas as pd
import os
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import pickle



def evaluate_model(y_test: pd.date_range, y_pred: pd.date_range) -> dict:
    try:
        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1s = f1_score(y_test, y_pred)
        return {
            'accuracy': acc,
            'precision': pre,
            'recall': rec,
            'f1_score': f1s
        }
    except Exception as e:
        raise Exception(f"Error in evaluating model:{e}")
def main():
    # y_test and y_pred
    test_data_path = r"./data/processed/test_processed.csv"
    x_test = pd.read_csv(test_data_path).iloc[:,0:-1].values
    y_test = pd.read_csv(test_data_path).iloc[:,-1].values

    y_pred = pickle.load(open("model.pkl","rb")).predict(x_test)
    try:
        metrics = evaluate_model(y_test, y_pred)
        with open('reports/metrics.json','w') as file:
            json.dump(metrics,file,indent=4)
    except Exception as e:
        raise Exception (f"Error in main file occured:{e}")
if __name__ == "__main__":
    main()