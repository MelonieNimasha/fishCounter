import numpy as np
import DeepLearning as DL

def trainer(N,X_data,Y_data,alpha,number):
	for num in range number:
		W,b= DL.initialize(N)
		Z,A= DL.forward_prop(W,b,X_data,N)
		Cost=DL.Cost_function(Y_caps,Y_data)
		print(Cost)
		dW, db= DL.back_propagation(N,Z,A,W,Y_data)
		W,b=DL.gradient_descent(W,b,dW, db,alpha)
	return W,b

def predictor(X,W,b):
	Z,A=DL.forward_prop(W,b,X_data,N)
	return A[str(len(N)-1)]

