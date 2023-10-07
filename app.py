import mlflow
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

import mlflow.sklearn

if __name__ == '__main__':
     x= np.array([-2,-4,-5,-6,0,1,2,4,5,8,7,2,4,8]).reshape(-1,1)
     y = np.array([1,1,1,0,1,1,0,0,0,1,1,1,0,0])

     rf = RandomForestClassifier()
     rf.fit(x, y)
     score = rf.score(x,y)
     print(score)
     mlflow.log_metric("score", score)

     mlflow.sklearn.log_model(rf,"Model")
     print('Model saved in run %s' %mlflow.active_run().info.run_uuid)