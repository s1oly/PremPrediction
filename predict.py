import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib


def from_csv(csv = None):
    df = pd.read_csv(csv)
    return df

