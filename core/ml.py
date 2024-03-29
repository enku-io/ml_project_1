import tensorflow as tf
import numpy as np # linear algebra
import pandas as pd
from core.preprocess import load_dataset
from sklearn import metrics

global model

def train_model(X_train, X_test, y_train, y_test):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(12, activation=tf.nn.relu, input_shape=[len(X_train.keys())]),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=100)
    model.save('my_model.h5')
    l, a = model.evaluate(X_test, y_test)
    print("Accuracy: ", a)

def predict_model(X_test,y_test):
    nn_y_pred = model.predict(X_test)

    nn_fpr, nn_tpr, thresholds = metrics.roc_curve(y_test, nn_y_pred)
    print("AUC: ", metrics.auc(nn_fpr, nn_tpr))
