import numpy as np

def initialize(N): #N= the list of number of hidden units in each layer
	W={}
	b={}
	L=len(N) #get the total number of layers
	for l in range(L-1):
		W[str(l+1)]=np.random.randn(N[l+1],N[l])*0.01 #random initiation of parameters
		b[str(l+1)]=np.zeros((N[l+1],1))			
	return W,b
	

def sigmoid(z):            #sigmoid activation  function
	return (1/(1+np.exp(z)))

def sigmoid_dash(Z):
	return(sigmoid(Z)*(1-ML.sigmoid(Z)))

def relu(z):
	z_abs=np.abs(z)    #relu activation function 
	return((z_abs+z)*0.5)

def forward_prop(W,b,X_data,N):
	m=(len(X_data)) #number of training examples
	n=(len(X_data[0])) #number of features in 0th layer
	N[0]=X_data.shape[0] #first element of N should be Features
	L=len(N) #number of units in each layer
	Z={} #cache dictionary
	A={} #activation dictionary
	A[str(0)]=X_data
	for l in range(L-1):
		Z[str(l+1)]=np.matmul(W[str(l+1)],A[str(l)])+b[str(l+1)] #Compute Z/cache
		if l<(L-2):
			A[str(l+1)]=relu(Z[str(l+1)]) #Compute activation for hidden layers
		else:
			A[str(l+1)]=sigmoid(Z[str(l+1)]) #Compute activation for outmost layer
	return Z,A

def Cost_function(Y_caps,Y_data):
	m=Y_data.shape[0] #get the number of labels
	Cost= (-(1/m))*np.sum(((Y_data*(np.log(Y_caps)))+((1-Y_data)*(np.log(1-Y_caps))))) #Compute cost
	return Cost

def back_propagation(N,Z,A,W,Y_data): 
	dA={}
	dZ={}
	dW={}
	db={}
	m=Y_data.shape[0]
	L=len(N)-1
	dA[str(L)]=np.sum((-(Y_data/A[str(L)]))+((1-Y_data)/(1-A[str(L)])))
	for l in range(L,0,-1):
		A_c = A[str(l-1)]
		A_t = A_c.transpose()
		dZ[str(l)]=dA[str(l)]*(ML.sigmoid_dash(Z[str(l)]))
		dW[str(l)]=(1/m)*(np.matmul(dZ[str(l)],A_t))
		db[str(l)]=(1/m)*(np.sum(dZ[str(l)], axis=1 ,keepdims=True))
		dA[str(l-1)]=np.matmul(((W[str(l)]).transpose()),dZ[str(l)])
	return dW, db

def gradient_descent(W,b,dW, db,alpha):
	W=W-alpha*dW
	b=b-alpha*db
	return W,b
