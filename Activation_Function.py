import numpy as np 

#Funcion de activacion Sigmoide 
def Sigmoid():
    return (Sigmoid_Function,Diff_Sigmoid)

def Sigmoid_Function(x):
    return 1/(1+np.exp(-x))

def Diff_Sigmoid(x):
    return Sigmoid_Function(x)*(1-Sigmoid_Function(x)) 
   
#Fucion de activacion tangente hiperbolica 
def Tanh():
    return (Tanh_Function,Diff_Tanh)
def Tanh_Function(x):
    return np.tanh(x)

def Diff_Tanh(x):
    y = ((1+Tanh_Function(x))*
    (1-Tanh_Function(x)))
    return y


