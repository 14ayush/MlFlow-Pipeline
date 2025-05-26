import mlflow
import mlflow.data
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
from sklearn.model_selection import GridSearchCV
import pandas as pd
import dagshub


data=load_breast_cancer()

x=pd.DataFrame(data.data,columns=data.feature_names)
y=pd.Series(data.target,name='target')

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)
rf=RandomForestClassifier(random_state=42)

#defining the parameters for grid search CV
params_grid={
    'n_estimators':[10,50,100,200,300],
    'max_depth':[None,10,30,50,100,200]
}

#applying the grid search cv
grid_search=GridSearchCV(estimator=rf,param_grid=params_grid,cv=5,n_jobs=-1,verbose=2)
#fit the model without MLFlow

'''

grid_search.fit(X_train,y_train)

#displaying the best params
best_parameters=grid_search.best_params_
bes_scorecard=grid_search.best_score_

print('best parameters',best_parameters)
print('best_score',bes_scorecard)
'''

#APPLYING WITH MLFLOW
mlflow.set_experiment('breast cancer dummy')
#this given particular code give only the best params what if we want all the parameters
'''with mlflow.start_run():
    grid_search.fit(X_train,y_train)
    best_parameters=grid_search.best_params_
    bes_scorecard=grid_search.best_score_
#log params
    mlflow.log_params(best_parameters)

#log training data

    train_df=X_train.copy()
    train_df['target']=y_train

    train_df=mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_df,'training')

#log test data

    test_df=X_test.copy()
    test_df['target']=y_test

    test_df=mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_df,'test data')

    mlflow.log_artifact(__file__)

    mlflow.sklearn.log_model(grid_search.best_estimator_,'random forest')

print(best_parameters)
print(bes_scorecard)
'''
grid_search.fit(X_train,y_train)

#code for the child parameters

with mlflow.start_run() as parent:
    for i in range(len(grid_search.cv_results_['params'])):
        with mlflow.start_run(nested=True) as child:
            mlflow.log_params(grid_search.cv_results_["params"][i])
            mlflow.log_metric("accuracy",grid_search.cv_results_["mean_test_score"][i])

best_parameters=grid_search.best_params_
bes_scorecard=grid_search.best_score_
mlflow.autolog()
print('best parameters',best_parameters)
print('best_score',bes_scorecard)