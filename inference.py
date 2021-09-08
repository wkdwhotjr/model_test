##################
# import package #
##################

import os
import json
import pandas as pd
import pickle as pk
from pathlib import Path

###################
# Set global Vars #
###################

MODEL_PATH = "../model/"

#############
# Functions #
#############

def load_model(model_path = MODEL_PATH):
    """
        sort model directory file by date and select recent model
    """
    recent_model = sorted(Path(model_path).iterdir(), key=os.path.getmtime)[0]
    model_path = os.path.join(model_path, recent_model)
    model = pk.load(model_path)

    return model


def inference(X_df : pd.DataFrame):
    """
        가장 최근에 저장되거나 수정된 모델을 불러오고 input에 따라 예측결과를 json으로 return 합니다.
    """
    model = load_model()
    pred = {"value" : model.predict(X_df)}
    return json.dump(pred)

