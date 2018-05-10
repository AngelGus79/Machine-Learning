# USAGEx
# python gd_dl.py

# import the necessary packages

# La idea general del uso de descenso de gradiente en machine learning
# es la siguiente:  
# Consiste en encontrar un modelo matematico lineal el cual pueda 
# predecir una clase (pred) mediante ciertas entradas (X = {x1, x2, x3 ...}) 
# el modelo tiene la siguiente forma w1x1 + w2x2 + w3x3 = pred
# por lo que la el trabajo consiste en encontrar las w's (representadas por W)
# El calculo de esta W, lo hacemos mediante un conjunto de datos(datos de entrenamiento).
# Es decir un conjunto de entradas x's  con sus respectivas clases y's.
# para esto seguimos el siguiente algoritmo general.
# 1) inicializamos las w's de manera aleatoria
# 2) calculamos el error que tiene el modelo con estas w's aleatorias
# en cada ciclo calculamos el error, evaluando todos los puntos.
# 3) se realizan los ajustes necesarios para minimizar el error. Generando
# nuevos w's. Se usa un tamaño de paso definido por alpha para disminuir
# el error.

import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import argparse

def sigmoid_activation(x):
	# compute and return the sigmoid activation value for a
	# given input value
        # devuelve un numero entre cero y uno
	return 1.0 / (1 + np.exp(-x))

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100,
	help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01,
	help="learning rate")
args = vars(ap.parse_args())

# generate a 2-class classification problem with 250 data points,
# where each data point is a 2D feature vector
# make_blobs, genera datos de manera aleatoria, n_samples es el numero de
# muestras que queremos generar. n_features indica la cantidad
# de dimensiones, o variables que queremos manejar. n_features = 2
# indica que tendremos x1 y x2. la variable 'y' es la salida
# representa 0 o 1.
(X, y) = make_blobs(n_samples=250, n_features=2, centers=2,
	cluster_std=1.05, random_state=20)
print (X)
print (y)

# concatenation of a column of 1's as the first entry in the feature
# vector -- this is a little trick that allows us to treat
# the bias as a trainable parameter *within* the weight matrix
# rather than an entirely separate variable

# para manejar de manera conjunta la variable b. (y = w1x1 + w2x2 + b)
# con los pesos que se van a calcular. se agrega en la matriz X
# (que son los datos de aprendizaje) una columna de 1's 

X = np.c_[np.ones((X.shape[0])), X]

# initialize our weight matrix such it has the same number of
# columns as our input features
# Se inicializan de manera aleatoria los pesos W
print("[INFO] starting training...")
W = np.random.uniform(size=(X.shape[1],))

# initialize a list to store the loss value for each epoch
lossHistory = []

# loop over the desired number of epochs

# itera un numero de veces finito ya que se podria ciclar
# si se intenta encontrar el minimo error del modelo
# puesto que el tamaño del paso que da podria ser
# muy largo

# En este ciclo se calcula el  error del modelo
# en un ciclo; se toman en cuenta todos los datos
# de aprendizaje X y 'y' y de alguna manera se obtiene
# la diferencia de la prediccion generada (pred)
# con la prediccion correcta (y)  (contenida en los datos de aprendizaje)
# y posteriormente se realizan ajustes en base en esta diferencia.

for epoch in np.arange(0, args["epochs"]):
# take the dot product between our features `X` and the
# weight matrix `W`, then pass this value through the
# sigmoid activation function, thereby giving us our
# predictions on the dataset
	preds = sigmoid_activation(X.dot(W))
	print('preds', preds)
# now that we have our predictions, we need to determine
# our `error`, which is the difference between our predictions
# and the true values

# aqui se obtiene la diferencia de la prediccion con lo correcto.
# preds = {0,1}  y = {0,1}, el resultado (error) es una lista
# de la evaluacion de cada una de las x's en X menos el resultado que debe ser
	error = preds - y
# given our `error`, we can compute the total loss value as
# the sum of squared loss -- ideally, our loss should
# decrease as we continue training

# esto es parte de la formula del calculo del error.
# Aqui se esta calculando la perdida, la cual queremos minimizar
	loss = np.sum(error ** 2)
	lossHistory.append(loss)
	print("[INFO] epoch #{}, loss={:.7f}".format(epoch + 1, loss))
# the gradient update is therefore the dot product between
# the transpose of `X` and our error, scaled by the total
# number of data points in `X`

# aqui empieza el calculo del ajuste que se va a aplicar en los pesos
# para minimizar el error.
# recuerda que X es una matriz de 2x250 y error es una matriz de
# 250x1.Osea se multiplican todas las x1's por la lista de errores
# y se suman estas multiplicaciones. la cual es la primer entrada  
# del resultado (el resuldo es una matriz de 2x1)
# luego se multiplican todas las x2's por la lista de errores la cual es la
# segunda entrada del resultado y finalmente se divide por 250
	gradient = X.T.dot(error) / X.shape[0]
# in the update stage, all we need to do is nudge our weight
# matrix in the opposite direction of the gradient (hence the
# term "gradient descent" by taking a small step towards a
# set of "more optimal" parameters

# Con el gradiente calculado en la instruccion anterior se realizan
# el ajuste en los pesos, el cual se multiplica por un alfa
# que es el tamaño del paso.
	W += -args["alpha"] * gradient

# to demonstrate how to use our weight matrix as a classifier,
# let's look over our a sample of training examples
for i in np.random.choice(250, 10):
# compute the prediction by taking the dot product of the
# current feature vector with the weight matrix W, then
# passing it through the sigmoid activation function
	activation = sigmoid_activation(X[i].dot(W))
# the sigmoid function is defined over the range y=[0, 1],
# so we can use 0.5 as our threshold -- if `activation` is
# below 0.5, it's class `0`; otherwise it's class `1`
	label = 0 if activation < 0.5 else 1
# show our output classification
	print("activation={:.4f}; predicted_label={}, true_label={}".format(
		activation, label, y[i]))

# compute the line of best fit by setting the sigmoid function
# to 0 and solving for X2 in terms of X1
Y = (-W[0] - (W[1] * X)) / W[2]

# plot the original data along with our line of best fit
plt.figure()
plt.scatter(X[:, 1], X[:, 2], marker="o", c=y)
plt.plot(X, Y, "r-")

# construct a figure that plots the loss over time
fig = plt.figure()
plt.plot(np.arange(0, args["epochs"]), lossHistory)
fig.suptitle("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()

