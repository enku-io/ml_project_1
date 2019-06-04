import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os
from app import ROOT


def load_dataset():
    df = pd.read_csv(os.path.join(ROOT,"data/insurance.csv")
    df = transform_features(df)
    df = encode_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    return X_train, X_test, y_train, y_test

def simplify_ages(data):
    bins = (15, 20, 30, 40, 50, 60, 70)
    group_names = ['Teenager', 'Twenties', 'Thirties', 'Fourties', 'Fifties', 'Sixities']
    categories = pd.cut(data.age, bins, labels=group_names)
    data.age = categories
    return data

def simplify_bmi(data):
    bins = (15, 25, 35, 45, 60)
    group_names = ['Small', 'Medium', 'Big', 'vBig']
    categories = pd.cut(data.bmi, bins, labels=group_names)
    data.bmi = categories
    return data

def encode_features(data):
    features = ['age', 'sex', 'bmi', 'smoker', 'region']

    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(data[feature])
        data[feature] = le.transform(data[feature])
    return data

def categorize(data):
    data['approved'] = 0

    for index, row in data.iterrows():
        if row['charges'] < 16639.912515:
            data.at[index, 'approved'] = 1
        else:
            data.at[index, 'approved'] = 0
    return data

def split_data(data):

    X_all = data.drop(['approved', 'charges'], axis=1)
    y_all = data['approved']

    num_test = 0.20
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=12)

    return X_train, X_test, y_train, y_test

def transform_features(data):
    data = simplify_ages(data)
    data = simplify_bmi(data)
    return data