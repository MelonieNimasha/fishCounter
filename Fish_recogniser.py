import numpy as np
import ML

def trainer(N,X_data,Y_data,alpha,number):
	for num in range number:
		W,b= initialize(N)
		Z,A= forward_prop(W,b,X_data,N)
		Cost=Cost_function(Y_caps,Y_data)
		print(Cost)
		dW, db= back_propagation(N,Z,A,W,Y_data)
		W,b=gradient_descent(W,b,dW, db,alpha)
	return W,b

def predictor(X,W,b):
	Z,A=forward_prop(W,b,X_data,N)
	return A[str(len(N)-1)]

