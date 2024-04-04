import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from warpdrive import WarpDrive


wd = WarpDrive()

df = wd.get_args("df")
# your code here
clf = DecisionTreeClassifier(random_state=0).fit(df.drop(labels='capsule', axis=1).values, df['capsule'].values)

col=df.drop(columns='capsule').columns.tolist()

wd.create_model(clf,"sklearn","DecisionTreeClassifier","df",col,"capsule")
