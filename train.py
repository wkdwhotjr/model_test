##################
# import package #
##################

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import Ridge, Lasso, LinearRegression, ElasticNet
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import pandas as pd
import numpy as np

import pickle as pk
import os

###################
# Set Global Vars #
###################

DATA_DIR = "../data/processed/"
MODEL_SAVE_DIR = "../model/"

CLF_MODELS = {
    "RF" : RandomForestClassifier(),
    "XGB" : XGBClassifier(),
    "LGBM" : LGBMClassifier(),
    "LR" : LogisticRegression(), 
    "ERT" : ExtraTreesClassifier(),
    "ADA" : AdaBoostClassifier(learning_rate=0.01),
    "DT" : DecisionTreeClassifier(),
}

REG_MODELS = {
    "RF" : RandomForestRegressor(),
    "XGB" : XGBRegressor(),
    "LGBM" : LGBMRegressor(),
    "RIDGE" : Ridge(),
    "LASSO" : Lasso(),
    "LR" : LinearRegression(),
    "EL" : ElasticNet(),
}

MODELS = {
    "reg" : REG_MODELS,
    "clf" : CLF_MODELS,
}

#############
# Functions #
#############

# Data loader
"""
    적절한 data_path를 지정해 주세요
"""
def load_data(data_path=os.path.join(DATA_DIR, "example.csv")):
    return pd.read_csv(data_path)


# Validation for result
"""
    각 문제마다 모델에 데이터가 적용 되었는지에 대한 metric을 담아 dict type으로 return 합니다.
    추후 문제별로 필요한 Metric이 있다면 추가해 주세요 ex) auc score, multi-class f1 score, MLE 등등
"""

def validation(model, X_df : pd.DataFrame , y_df : pd.DataFrame, mode : str):

    real = y_df
    pred = model.predict(X_df)

    scores = {}

    if mode == 'clf':
        scores['Accuracy'] = accuracy_score(real, pred)
        scores['Precision'] = precision_score(real, pred)
        scores['Recall'] = recall_score(real, pred)
        scores['F1-Score'] = f1_score(real, pred)
        scores['ConfusionMatrix'] = confusion_matrix(real,pred)
    
    elif mode == 'reg':
        scores['MSE'] = mean_squared_error(real, pred)
        scores['MAE'] = mean_absolute_error(real, pred)
        scores['R2_Score'] = r2_score(real, pred)

    return scores

# Save model for pickle file
"""
    pickle 파일의 형태로 model을 저장합니다.
    추후 다른 확장자에 대해서도 extension을 통해 설계해 주세요.
"""
def model_save(model, name):
    """
        model save as pickle extension
    """
    extension = ".pickle"

    save_model_name = os.path.join(MODEL_SAVE_DIR, name + extension)
    pk.dump(model, save_model_name)
    

# Train model
"""
    ML 모델을 훈련시킵니다. 
"""
def train(X : list, y : list, model_name : str, data, mode,*vals,**args):
    """
        Train model for specific problems ( Regression, Classification )

        X : list, X features for fitting model in data

        y : list, target value for prediction

        model_name : str, scikit-learn machine learning model or some models that have "fit" method
            - reg : { RF, XGB, LGBM, RIDGE, LASSO, LR, EL }
            - clf : { RF, XGB, LGBM, LR, ERT, ADA, DT }

        data : pd.DataFrame, data that fitting to AI

        mode : str, { "reg" , "clf" }, proper type for solving problem reg means regression, clf means classification
    """
    # 훈련 결과 dictionary
    scores = {}

    # 훈련 모드 선택

    model = MODELS[mode][model_name]

    X_train, X_val, y_train, y_val = train_test_split(data[X], data[y], train_size=0.7)
    
    model.fit(X_train, y_train)

    scores['train'] = validation(model, X_train, y_train, mode)
    scores['validation'] = validation(model, X_val, y_val, mode)

    return scores


########
# main #
########

if __name__ == "__main__":

    #Load data
    data = load_data()
    
    #Set Features
    """
        각 프로젝트에 맞도록 EDA를 수행하고, 그 결과 모델이 적합한 X 인자 column 이름을 넣어 주세요
        Target은 예측이 필요한 column 이름 입니다.
            - Multi Output을 지원 하지만 추천하지 않습니다 단일 값이 들어 있는 list로 넣어 주세요
            - 필요할 경우 Multi-output을 코드에 맞게 사용해 주세요
    """
    X_features = []
    target = []

    #Set evaluation mode
    """
        reg, clf 두가지 키워드 중 앎맞은 키워드를 넣어주세요, reg는 회귀 문제, clf는 분류 문제에서 선택 하시면 됩니다. 
    """
    mode = "reg" 
    models = MODELS[mode]

    # initiate metric
    """
        필요한 Metric에 따라 코드를 수정 사용해 주세요

        기본값(Default) :
            회귀 : MAE(Mean Absoulte Error, L1 Norm)
            분류 : Accuracy
        
    """
    selected_model = None
    accuracy = None
    mae = np.inf

    # fitting model and finding optimal model
    """
        Scikit-Learn의 fit method가 존재하는 모든 BaseModel들을 fitting 합니다. 
    """
    for model_name, model in models.items():
        print(f"Model, {model_name} train start")
        result = train(X_features, target, model_name, data, mode)
        
        for score_name, train_score in result['train'].items():
            print("--Train Score--")
            print(f"    {score_name} : {train_score}")

        for score_name, val_score in result['validation'].items():
            print("--Validation Score--")
            print(f"    {score_name} : {val_score}")

        # 데이터가 많아서 validation이 가능할 경우, result의 키 값을 trian이 아니라 validation으로 바꿔 사용 해 주세요

        if mode == "clf":
            running_acc = result["Accurcy"]
            if running_acc > accuracy :
                accuracy = running_acc
                selected_model = model_name

        elif mode == "reg":
            running_mae = result['MAE']
            if running_mae < mae:
                mae = running_mae
                selected_model = model_name

    # Save model
    """
        가장 좋았던 모델을 pickle 파일로 저장합니다. string 타입으로 2번째 인자에 이름을 넣어주세요.
        이름을 입력할 때 필요한 적절한 Convention(ex. 회사이름명_날짜_모델번호, 책임자이름명_모델명_번호 등)을 분석팀 여러분이 정해 주세요, 개발팀과 협의 해 보아도 좋습니다.
    """
    model_save(MODELS[mode][selected_model], "optimal1") # 모델 이름이 곂치지 않게 이름을 정해 주세요

        

        


