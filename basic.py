from flask import Flask, render_template
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix, accuracy_score
import io
from sklearn.preprocessing import MinMaxScaler



app = Flask(__name__)


@app.route('/')
def home():

	# ANN

	dataset_test=pd.read_csv('Google_Stock_Price_Test.csv')
	X_test=dataset_test.iloc[:-1,1:2].values 
	y_test=dataset_test.iloc[1:,1:2].values
	ann = tf.keras.models.load_model('ann.h5')
	y_pred = ann.predict(X_test)
	fig = Figure()
	plt = fig.add_subplot(1, 1, 1)
	plt.plot(y_test, color = 'red', label = 'Real Google Stock Price')
	plt.plot(y_pred, color = 'blue', label = 'Predicted Google Stock Price')
	plt.set_title('Google Stock Price Prediction')
	plt.set_xlabel('Time')
	plt.set_ylabel('Google Stock Price')
	fig.savefig('static/img/annGraph.png')
	# plt.legend()
	# plt.show()
	# Convert plot to PNG image
	pngImage = io.BytesIO()
	FigureCanvas(fig).print_png(pngImage)
	# Encode PNG image to base64 string
	pngImageB64String = "data:image/png;base64,"
	pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')

	# RNN1
	regressor = tf.keras.models.load_model('rnn_old.h5')
	dataset_train=pd.read_csv('Google_Stock_Price_Train.csv')
	trainig_set=dataset_train.iloc[:,1:2].values  #creating a numarray that contains the open price of the stock

	"""### Feature Scaling"""

	from sklearn.preprocessing import MinMaxScaler
	sc=MinMaxScaler(feature_range=(0,1))  #all stock prices will be between 0 and 1
	training_set_scaled=sc.fit_transform(trainig_set)
	dataset_train=pd.read_csv('Google_Stock_Price_Train.csv')
	dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
	real_stock_price = dataset_test.iloc[:, 1:2].values

	"""### Getting the predicted stock price of 2017"""

	dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
	inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
	inputs = inputs.reshape(-1,1)
	
	inputs = sc.transform(inputs)
	X_test = []
	for i in range(60, 80):
		X_test.append(inputs[i-60:i, 0])
	X_test = np.array(X_test)
	X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
	predicted_stock_price = regressor.predict(X_test)
	predicted_stock_price = sc.inverse_transform(predicted_stock_price)
	#plotting the figure
	fig_rnn = Figure()
	plt = fig_rnn.add_subplot(1, 1, 1)
	plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
	plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
	plt.set_title('Google Stock Price Prediction')
	plt.set_xlabel('Time')
	plt.set_ylabel('Google Stock Price')
	fig_rnn.savefig('static/img/RNN1Graph.png')

	# RNN2
	regressor = tf.keras.models.load_model('rnn_new.h5')
	dataset_train=pd.read_csv('Google_Stock_Price_Train.csv')
	my_train=dataset_train.iloc[:,1:]  #creating a numarray that contains the open price of the stock
	training_set=my_train.replace(",","",regex=True)
	"""### Feature Scaling"""

	from sklearn.preprocessing import MinMaxScaler
	sc=MinMaxScaler(feature_range=(0,1))  #all stock prices will be between 0 and 1
	training_set_scaled=sc.fit_transform(training_set)
	print(training_set_scaled)

	"""### Creating a data structure with 60 timesteps and 1 output

	Timesteps:Data structure specifying what the rnn needs to remeber.
	"""

	X_train=[]
	y_train=[]
	for i in range(60,1258): #we will select the first 60 values to predict the first value and so on
		X_train.append(training_set_scaled[i-60:i])  #will put first 60 value in x
		y_train.append(training_set_scaled[i,0])#will put the value we will predict
	X_train,y_train=np.array(X_train),np.array(y_train)
	dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
	real_stock_price = dataset_test.iloc[:,1:2].values
	data_set1=dataset_train.iloc[:,1:]
	data_set2=dataset_test.iloc[:,1:]
	dataset_total = pd.concat((data_set1, data_set2), axis = 0)
	dataset_total=dataset_total.replace(",","",regex=True)
	inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
	inputs = sc.transform(inputs)
	X_test = []
	for i in range(60, 80):
		X_test.append(inputs[i-60:i])
	X_test = np.array(X_test)
	# # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
	predicted_stock_price = regressor.predict(X_test)
	# predicted_stock_price = sc.inverse_transform(predicted_stock_price)
	predicted_stock_price=predicted_stock_price*(1/1.86025746e-03)
	predicted_stock_price=predicted_stock_price+300
	#plotting the figure
	fig_rnnNew= Figure()
	plt = fig_rnnNew.add_subplot(1, 1, 1)
	plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
	plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
	plt.set_title('Google Stock Price Prediction')
	plt.set_xlabel('Time')
	plt.set_ylabel('Google Stock Price')
	fig_rnnNew.savefig('static/img/RNN2Graph.png')


	return render_template('after.html')


if __name__ == "__main__":
    app.run(debug=True)