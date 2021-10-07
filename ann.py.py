import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

dataset_test=pd.read_csv('Google_Stock_Price_Test.csv')

X_test=dataset_test.iloc[:-1,1:2].values 
y_test=dataset_test.iloc[1:,1:2].values

ann = tf.keras.models.load_model('ann.h5')

y_pred = ann.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
plt.plot(y_test, color = 'red', label = 'Real Google Stock Price')
plt.plot(y_pred, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
