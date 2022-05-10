import numpy as np
import random 


class functions_utils  : 
	def sigmoid(z):
	  return 1.0 / (1 + np.exp(-z))# Derivative of sigmoid function
	def sigmoid_prime(z):
	  return sigmoid(z) * (1-sigmoid(z))
	#****************************************
	def tanh(z):
		return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))# Derivative of Tanh Activation Function
	def tanh_prime(z):
		return 1 - np.power(tanh(z), 2)
	#****************************************
	def relu(z):
	  return max(0, z)# Derivative of ReLU Activation Function
	def relu_prime(z):
	  return 1 if z > 0 else 0
	#****************************************
	def softmax(z):
		e = np.exp(z)
		return e / e.sum()



class Layer : 
	def __init__(self ,prev_n ,  n  , activation = "sigmoid" ,w = None , b = None) : 
		self.w = np.array([[random.uniform(0, 1) for i in range(0,prev_n)] for j in range(n )]) if w == None else np.array(w)
		self.b = np.array([random.uniform(0, 1) for j in range(n )]) if b == None else np.array(b)
		self.out = None 
		self.act = activation
		assert activation in ["relu" , "soft" , "tanh", "sigmoid"]

	def set_Last_Layer(self) : 
		self.act = "soft"


	def forward(self,x) : 
		self.out = self.w.dot(x) + self.b 	
		if self.act == "relu" : 
			self.out = functions_utils.relu(self.out)
		if self.act == "soft"  : 
			self.out = functions_utils.softmax(self.out)
		if self.act == "tanh"  : 
			self.out = functions_utils.tanh(self.out)
		if self.act == "sigmoid"  : 
			self.out = functions_utils.sigmoid(self.out)
		return self.out


		



class Cnn : 
	def __init__(self,*args) : 
		self.dims = list(args)
		self.t_dims = [(self.dims[i] , self.dims[i+1]) for i in range(len(self.dims) - 1 )]
		#self.layers = [Layer(*i , activation = "tanh") for i in self.t_dims]
		self.layers = [
			Layer(2,2 , w = [[20.0, -20.0],[-20.0,20.0]]  , b = [-10.0 , 30.0]) , 
			Layer(2 , 1 , w = [[20.0,20.0] ] , b = [-30.0])

		]
		#self.layers[-1].set_Last_Layer()
		self.labels = [1,2,3] 

	def setLabels(self,maplabels = list()) : 
		self.labels = maplabels 


	def calculate(self,x) : 
		y = x.copy()
		for i in range(len(self.layers)) : 
			y = self.layers[i].forward(y)
			print(i,y)
		return y

"""
c = Layer(5,6,"soft")
c.forward(np.array([random.uniform(0, 1) for i in range(0,5)]))
print(c.out)
"""

c = Cnn(2,2,1)
print(c.calculate(np.array([1.0 ,1.0 ])))
