import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import certifi

# Set SSL_CERT_FILE to certifi's CA bundle
os.environ["SSL_CERT_FILE"] = certifi.where()

import dagshub
#Resolving the SSL error 

#settng up the dagshub so that we can see the experiments on the remote server (dagshub)
dagshub.init(repo_owner='14ayush', repo_name='MlFlow-Pipeline', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/14ayush/MlFlow-Pipeline.mlflow")
mlflow.set_experiment('My First Experiment')


wine=load_wine()
x=wine.data
y=wine.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)

max_depth=10
n_estimators=25
#setting up the new experiment



with mlflow.start_run():
    rf=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators)
    rf.fit(x_train,y_train)

    y_pred=rf.predict(x_test)
    accuracy=accuracy_score(y_test,y_pred)

    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_param('max_depth',max_depth)
    mlflow.log_param('n_estimators',n_estimators)
    #creating the confusion matrix

    cm=confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(7,7))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=wine.target_names,yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('prediction')
    plt.title('confusion_matrix')
    plt.savefig("confusion_matrix.png")

    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)

    #log the tags
    mlflow.set_tags({"Author":'Ayush sharma',"Project":'Wine Classification'})
    mlflow.sklearn.log_model(rf,"Random Forest Classification")

    print(accuracy)
