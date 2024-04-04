import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from warpdrive import WarpDrive


wd = WarpDrive()

df = wd.get_args("df")
# your code here

# X, y = load_iris(return_X_y=True)
# clf = LogisticRegression(random_state=0).fit(X, y)
# clf.predict(X[:2, :])
# clf.predict_proba(X[:2, :])
# clf.score(X, y)
clf = LogisticRegression(random_state=0).fit(df.drop(labels='capsule', axis=1).values, df['capsule'].values)
col=df.drop(columns='capsule').columns.tolist()
wd.create_model(clf,"sklearn","LogisticRegressionClassifier","df",col,"capsule")
