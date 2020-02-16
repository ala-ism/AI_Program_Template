#%%
import pandas as pd
import numpy as np

#%%
#Defining the sigmoid function (some of you may know it as the logistic function) and its derivative
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

#%%
#The loss function that will be used in this example is simple the mean squared error: Loss(Y,Ypred)=sum(Y-Ypred)^2
def Loss(Y,Ypred):
    loss=0
    for i in range(len(Y)):
        loss+=(Y[i]-Ypred[i])**2
    return loss

#%%
#network initialization
class MyNeuralNet:
    def __init__(self,x,y):
        self.input=x
        self.weights1=np.random.rand(self.input.shape[1],4)
        self.weights2=np.random.rand(4,1)
        self.y=y
        self.output=np.zeros(y.shape)

    #feedforward 
    def feedforward(self):
        self.layer1=sigmoid(np.dot(self.input, self.weights1))
        self.output=sigmoid(np.dot(self.layer1, self.weights2))
        
    #back propagation
    def backprop(self):
        #loss function derivative with respect to W1 and W2
        d_weights2=np.dot(self.layer1.T,(2*(self.y-self.output)*sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        #updating the weights with the loss function derivative
        self.weights1 += d_weights1
        self.weights2 += d_weights2

#%%
#Let's try it out
if __name__=="__main__":
    X=np.array([[0,0,1], [0,1,1], [1,0,1],[1,1,1]])
    y=np.array([[0],[1],[1],[0]])
    nn=MyNeuralNet(X,y)

    for i in range(15000):
        nn.feedforward()
        nn.backprop()
    
    print(f"prediction is:\n{nn.output}\n")
    print(f"loss is:\n{Loss(nn.y,nn.output)}")

