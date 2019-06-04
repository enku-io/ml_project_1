from flask import Flask
import core.ml
import os
from core.ml import  train_model,predict_model
from core.preprocess import load_dataset


app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'

if __name__ == '__main__':
    # app.run()
    X_train, X_test, y_train, y_test = load_dataset()
    train_model(X_train, X_test, y_train, y_test)
    predict_model(X_test,y_test)