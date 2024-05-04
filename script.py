
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
import sklearn
import boto3
import pathlib
from io import StringIO
import argparse

# Load model
def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

# Train model
if __name__ =='__main__':
    # Parse arguments
    print("[INFO] Parsing arguments")
    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--random_state', type=int, default=0)

    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--train-file', type=str, default='train-V1.csv')
    parser.add_argument('--test-file', type=str, default='test-V1.csv')

    args, _ = parser.parse_known_args()

    print("Sklearn version: " + sklearn.__version__)
    print("Joblib version: " + joblib.__version__)
    
    print('[INFO] Reading data')
    print()
    # Load data
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))

    features = list(train_df.columns)
    label = features.pop(-1)

    print("Building training and testing datasets")
    print()
    # Split data
    X_train = train_df[features]
    y_train = train_df[label]
    X_test = test_df[features]
    y_test = test_df[label]

    print("Column Order:")
    print(features)
    print()
    print("lanel column:") 
    print(label)
    print()   
    print("---shape of training and testing datasets---")
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print()

    print("Trainning Random Forest model ...")
    print()
    model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state)
    model.fit(X_train, y_train)
    print()

    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print("Model saved at: {}".format(model_path))
    print()

    y_pred_test = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_report = classification_report(y_test, y_pred_test)

    print()
    print("---Metrics Results for Testing Data---")
    print()
    print("Total Rows are:", X_test.shape[0])
    print("[TESTING] Accuracy: ", test_accuracy)
    print("[TESTING] Classification Report: ")
    print(test_report)
    
