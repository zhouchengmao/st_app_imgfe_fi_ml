# coding: utf-8

import os  # 用于文件名和路径处理
import math

import numpy as np
import pandas as pd
##import matplotlib.pyplot as plt
##import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit, cross_val_score, KFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, roc_curve, \
    confusion_matrix, make_scorer, mean_squared_error, zero_one_loss, log_loss
from sklearn.impute import SimpleImputer
# ref: https://stackoverflow.com/questions/41993565/save-minmaxscaler-model-in-sklearn
from lightgbm import LGBMClassifier

# 模型初始化参数备份
# lr = LogisticRegression(penalty='l2',tol=0.0000001,C=100,fit_intercept=True,intercept_scaling=1,max_iter=100,multi_class='ovr',verbose=0,warm_start=False,n_jobs=1)  # 逻辑回归模型
# tr = DecisionTreeClassifier(splitter='best', max_depth=2, min_samples_split=50,min_samples_leaf =20, min_weight_fraction_leaf=0.001,  random_state=1)  # 决策树模型
# forest = RandomForestClassifier(n_estimators=10, min_samples_leaf = 15 , criterion='entropy', n_jobs = -1,random_state =1)  #　随机森林
# Gbdt = GradientBoostingClassifier(learning_rate=0.08,n_estimators=10,max_depth=2, max_features=3, min_samples_split=20,min_samples_leaf=3,random_state =1)  # gbdt
# Xgbc = XGBClassifier(learning_rate=0.12,n_estimators=12,  max_depth=2,  min_child_weight = 1,gamma=0.1,scale_pos_weight=1)  # Xgbc
# gbm = LGBMClassifier(learning_rate=0.11, n_estimators=10, lambda_l1=0.01 ,lambda_l2= 10 ,max_depth=2, bagging_fraction = 0.8,feature_fraction = 0.5)  # lgbm
# catb = CatBoostClassifier(learning_rate=0.11, n_estimators=10, max_depth=2) # catb


# joblib中有万能dump、load：适用于Imputer、Scaler、Model……
import joblib


# 保存模型到文件
def dump_obj(obj, obj_filename):
    return joblib.dump(obj, obj_filename)


# 从文件读取已保存模型
def load_obj(obj_filename):
    return joblib.load(obj_filename)


N_FEATS = 40  # 100


# 封装操作：特征选择(LightGBM模型-特征重要性)
# LightGBM-LGBMClassifier 利用特征重要性排序来选取出重要特征
def to_lgbm_fi(df, target_name='target', n_feats=N_FEATS):
    y_column = target_name
    X_columns = [col for col in df.columns if col != y_column]
    X = df[X_columns]
    y = df[y_column]

    model = LGBMClassifier()
    model.fit(X, y)

    feature_importance = model.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)

    new_cols = pd.Series(X_columns)[sorted_idx][-n_feats:].to_list()
    fis = feature_importance[sorted_idx][-n_feats:]
    print(new_cols, fis)
    new_cols = new_cols + [target_name]
    return df[new_cols], new_cols


# 读取csv数据
def read_csv(fp, encoding='gb18030'):
    pocd = pd.read_csv(fp, encoding=encoding)
    return pocd


# 通用的X, y列分割
def split_x_y(df):
    # 分割出 (变量列, 目标列)
    return df.iloc[:, 0:-1], df.iloc[:, -1:]


# 预处理数据（用于训练，有y）
def preprocessing_data_train(pocd):
    # 对缺失行进行处理
    pocd = pocd.dropna()

    X, y = split_x_y(pocd)

    # 插补
    imp = SimpleImputer(missing_values=np.NaN, strategy='mean')
    X = imp.fit_transform(X)

    # 标准化
    ss = StandardScaler()
    X = ss.fit_transform(X)

    # 恢复表头
    new_pocd = pd.concat([pd.DataFrame(X), y], axis=1)
    new_pocd.columns = pocd.columns
    pocd = new_pocd

    return pocd, imp, ss


# 预处理数据（用于纯预测，无y）
def preprocessing_data_predict(pocd, imp, ss, target_name='target'):
    # 对缺失行进行处理
    pocd = pocd.dropna()

    # 插补
    new_pocd = imp.transform(pocd)

    # 标准化
    new_pocd = ss.transform(new_pocd)

    # 恢复表头
    new_pocd = pd.DataFrame(new_pocd)
    new_pocd.columns = pocd.columns
    pocd = new_pocd

    return pocd


# 模型训练
def model_fit(model, X, y):
    return model.fit(X, y)


# 模型评分
def model_score(model, X, y):
    model_result = {}
    y_proba = model_result["y_proba"] = model.predict_proba(X)
    y_pre = model_result["y_pre"] = model.predict(X)
    score = model_result["score"] = model.score(X, y)
    acc_score = model_result["accuracy_score"] = accuracy_score(y, y_pre)
    preci_score = model_result["preci_score"] = precision_score(y, y_pre)
    rec_score = model_result["recall_score"] = recall_score(y, y_pre)
    f1__score = model_result["f1_score"] = f1_score(y, y_pre)
    auc = model_result["auc"] = roc_auc_score(y, y_proba[:, 1])

    mse = model_result["mse"] = mean_squared_error(y, y_pre)
    zero_one_loss_fraction = model_result["zero_one_loss_fraction"] = zero_one_loss(y, y_pre, normalize=True)
    zero_one_loss_num = model_result["zero_one_loss_num"] = zero_one_loss(y, y_pre, normalize=False)

    return model_result


# 模型预测
def model_predict(model, X):
    return model.predict(X)


# 模型预测（概率）
def model_predict_proba(model, X):
    return model.predict_proba(X)
