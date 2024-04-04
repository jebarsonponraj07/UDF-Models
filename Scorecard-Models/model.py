import xgboost as xgb
from xgboost import XGBClassifier
# from sklearn import XGBClassifier

clf = XGBClassifier(random_state=0).fit(z.drop(labels='capsule', axis=1).values, z['capsule'].values)
# print(df)
col=z.drop(columns='capsule').columns.tolist()
wd.create_model(clf,"xgboost","XGBoost_Classifier","z",col,"capsule")
