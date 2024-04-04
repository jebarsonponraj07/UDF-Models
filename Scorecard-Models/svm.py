import pandas as pd
import numpy as np
from sklearn import svm
from warpdrive import WarpDrive


wd = WarpDrive()

df = wd.get_args("df")
# your code here

clf = svm.SVC(random_state=0).fit(df.drop(labels='capsule', axis=1).values, df['capsule'].values)

col=df.drop(columns='capsule').columns.tolist()
wd.create_model(clf,"sklearn", "svm", "df",col,"capsule")
