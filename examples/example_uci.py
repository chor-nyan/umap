import wget
import pandas as pd
import xgboost as xgb
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import pprint

link_to_data = 'http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
ClevelandDataSet = wget.download(link_to_data)
print(ClevelandDataSet)

col_names = ['age','sex','cp','restbp','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num']

heart_data_df = pd.read_csv(ClevelandDataSet, sep=',', header=None, names=col_names, na_filter= True, na_values= {'ca': '?', 'thal': '?'})
heart_data_df.head()
heart_data_df = heart_data_df.dropna(how='any',axis=0)
heart_data_df['diagnosed'] = heart_data_df['num'].map(lambda d: 1 if d > 0 else 0)
feature_cols = ['age','sex','cp','restbp','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
features_df = heart_data_df[feature_cols]
heart_train, heart_test, target_train, target_test = \
    model_selection.train_test_split(features_df, heart_data_df.loc[:,'diagnosed'], test_size=0.33, random_state=0)

dm_train = xgb.DMatrix(heart_train, label=target_train)
dm_test = xgb.DMatrix(heart_test)

param = {'objective':'multi:softmax', 'max_depth':2, 'eta':0.8, 'num_class': 2, 'eval_metric': 'auc', 'silent':1 }
xgb_model = xgb.train(param, dm_train)
y_predict = xgb_model.predict(dm_test)
print(y_predict)

accuracy = accuracy_score(target_test, y_predict)
print("Accuracy: " +  str(accuracy))