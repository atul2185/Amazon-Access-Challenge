# -*- coding: utf-8 -*-
"""


@author: Z078739
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
# Function for checking the hyperparameter for Logistics regression
def logistics_param():
    alist=[]
    for i in range(1,20):
        lr=LogisticRegression(C=i)
        roc_auc=cross_val_score(lr,train,y,cv=10,scoring="roc_auc")
        mean_roc_auc=np.mean(roc_auc)
        adict={i:mean_roc_auc}
        alist.append(adict)
    check={k:v for d in alist for k,v in d.items()}
    max1=max(check.values())
    key=[k for k,v in check.items() if v==max1]
    value=key[0]
    return value
df=pd.read_csv("C:\\Users\\z078739\\Downloads\\amazon.csv")
test=pd.read_csv("C:\\Users\\z078739\\Downloads\\test.csv")
df.info()
test.info()
y=df.ix[:,0]
x=df.ix[:,1:]
test1=test.drop("id",1)
frame=x.append(test1,ignore_index=True)
encoder=OneHotEncoder()
frame1=encoder.fit_transform(frame)
frame1=frame1.toarray()
frame2=pd.DataFrame(frame1)
train=frame2.ix[0:32768,:]
test2=frame2.ix[32769:,:]
lr1=LogisticRegression(C=logistics_param())
lr1.fit(train,y)
predict_lr1=lr1.predict(test2)
predict_lr2=pd.DataFrame(predict_lr1)
predict_lr2.columns=["Prediction_LR"]
# predict_lr1.to_csv("C:\Users\z078739\Downloads\code_check.csv",sep="\t",)

# Model development process for Random Forest 
from sklearn.ensemble import RandomForestClassifier
def rf_param():
    alist=[]
    for i in range(100,900,100):
        rf=RandomForestClassifier(n_estimators=i)
        roc_auc=cross_val_score(rf,x,y,cv=10,scoring="roc_auc")
        mean_roc_auc=np.mean(roc_auc)
        adict={i:mean_roc_auc}
        alist.append(adict)
    check={k:v for d in alist for k,v in d.items()}
    max1=max(check.values())
    key=[k for k,v in check.items() if v==max1]
    value=key[0]
    return value
rf1=RandomForestClassifier(n_estimators=rf_param())
rf1.fit(x,y)
predict_rf1=rf1.predict(test1)
predict_rf2=pd.DataFrame(predict_rf1)
predict_rf2.columns=["Prediction_RF"]


# Model development process for Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
def gb_param():
    alist=[]
    for i in range(1,20):
        gb=GradientBoostingClassifier(max_depth=i)
        roc_auc=cross_val_score(gb,x,y,cv=10,scoring="roc_auc")
        mean_roc_auc=np.mean(roc_auc)
        adict={i:mean_roc_auc}
        alist.append(adict)
    check={k:v for d in alist for k,v in d.items()}
    max1=max(check.values())
    key=[k for k,v in check.items() if v==max1]
    value=key[0]
    return value
gb1=GradientBoostingClassifier(max_depth=gb_param())
gb1.fit(x,y)
predict_gb1=gb1.predict(test1)
predict_gb2=pd.DataFrame(predict_gb1)
predict_gb2.columns=["Prediction_GB"]

# Creation of ensempble classifier, voting will chose for deciding a final prediction
submission=pd.concat([predict_lr2,predict_rf2,predict_gb2],axis=1)
submission["sum1"]=submission["Prediction_LR"]+submission["Prediction_RF"]+submission["Prediction_GB"]
def ensemble(row):
    if row["sum1"]>=2:
        return 1
    return 0
submission["Action"]=submission.apply(lambda row: ensemble (row),axis=1)    
submission=submission.drop(["Prediction_LR","Prediction_RF","Prediction_GB","sum1"],1)
submission['id']=submission.index
cols=submission.columns.tolist()
cols=cols[-1:]+cols[:-1]
submission=submission[cols]
submission["id"]=submission["id"]+1
submission.to_csv("C:\Users\z078739\Downloads\submission.csv",sep="\t",index=False)

    





