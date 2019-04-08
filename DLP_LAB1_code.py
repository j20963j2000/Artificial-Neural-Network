#!/usr/bin/env python
# coding: utf-8

# In[1]:


def generate_linear(n = 100):
    import numpy as np
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    import numpy as np
    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)
        
        if 0.1*i == 0.5:
            continue
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
        
    return np.array(inputs), np.array(labels).reshape(21, 1)


X_linear, y_linear = generate_linear(n = 100)
X_XOR, y_XOR = generate_XOR_easy()
#print(X[:5])
#print(y[:5])
#print(X.head(5))
#print(y.head(5))


# In[2]:


import numpy as np
import pandas as pd
import math


# In[6]:


class Layer:   #先設計單層layer的架構，再輸入Neural Network當input
    def __init__(self, inputs, neurons, weights = None, activation = None, bias = None):
        
        self.weights = weights if weights is not None else np.random.randn(inputs, neurons)  #weight and bias 都用random來設定初始值
        self.activation = activation       #輸入activation function
        self.last_activation = None       #存取經過activation function的 output
        self.bias = bias if bias is not None else np.random.randn(neurons)   #weight and bias 都用random來設定初始值
        self.error = None
        self.delta = None                #存取loss function對該層layer的偏微
        
    def activate(self, X):         #function for node output
        #print(self.weights)
        r = np.dot(X, self.weights) + self.bias
        self.last_activation = self.activationapply(r)
        return self.last_activation
    
    def activationapply(self, r):   #function for activating
        
        if self.activation == None:
            return r
        
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-r))
        
        return r
    
    def activation_der(self, r):     #function for derivative function for sigmoid function
        
        if self.activation == None:
            return r
        if self.activation == 'sigmoid':
            return r * (1 - r)
    


# In[7]:


class NN:  #Neural Network and Backpropagation
    def __init__(self):
        
        self._layers = []      #將輸入的layer當作 inputs，並存到變數self._layers
    
    def add_layer(self, layer):  #加入幾層的layer當作input

        self._layers.append(layer)
        
    def forward(self, X):   #計算每一層layer node的output
         
        for layer in self._layers:
            X = layer.activate(X)
        
        return X
    
    def predict(self, X):  #predict output value 

        f = self.forward(X)
#        print(f.shape)參數試算
        f = np.where(f >= 0.5, 1, 0) #利用where()設置threshold值，會輸出一個True and False list，True值會輸出1，反之輸出0
        return f
        
    def backpropagation(self, X, y, learning_rate):
        
        output = self.forward(X)
        
        for i in reversed(range(len(self._layers))):   
            layer = self._layers[i]
            
            if layer == self._layers[-1]:
                layer.error = y - output
                layer.delta = np.multiply(layer.error, layer.activation_der(output))
            else:
                next_layer = self._layers[i + 1]
                layer.error = np.dot(next_layer.delta, next_layer.weights.T)
#                print(layer.error)
                layer.delta = layer.activation_der(layer.last_activation) * layer.error
#                print('-', layer.activationapply(layer.last_activation))
                
                
        for i in range(len(self._layers)):
            layer = self._layers[i]
            input_to_use = np.atleast_2d(X if i == 0 else self._layers[i - 1].last_activation)
#            print(layer.delta)
#            print(input_to_use)
            layer.weights += layer.delta * input_to_use.T * learning_rate
            learning_rate = learning_rate * 0.99
            
#            print('shape :', (layer.delta * input_to_use.T).shape)
#            print('weight:', layer.weights)
    
    def training(self, X, y, learning_rate, epochs):
        
        mse_total = []
        
        for i in range(epochs):
            for j in range(len(X)):
                self.backpropagation(X[j], y[j], learning_rate)

#                 if i > 80000:
#                     learning_rate = 0.5
#                 if self.accurary(y_pred = self.predict(X_XOR), y_true = y_XOR) == 1:
#                     print(self.accurary(y_pred = self.predict(X_XOR), y_true = y_XOR))
#                     break
                
            if i % 1000 == 0:
                    
                mse = np.mean(np.square(y - nn.forward(X)))
                mse_total.append(mse)
                
                print('epoch :', i)
                print('MSE :', sum(mse_total)/len(mse_total))
            
        return mse_total
    
    def accurary(self, y_pred, y_true):
        acc = np.mean(y_pred ==y_true)
        return acc
            


# In[10]:


nn = NN()
nn.add_layer(Layer(2, 4,activation = 'sigmoid'))
nn.add_layer(Layer(4, 4,activation = 'sigmoid'))
nn.add_layer(Layer(4, 1,activation = 'sigmoid'))

#X_train = np.array(X[:13])
#y_train = np.array(y[:13])

#X_test = np.array(X[13:])
#y_test = np.array(y[13:])
print('linear :')
fit = nn.training(X_linear, y_linear, 0.1, 10000)
prediction_linear = nn.predict(X_linear)
print('Acc :', nn.accurary(y_pred = nn.predict(X_linear), y_true = y_linear))
print('--------------')
print('XOR :')
fit = nn.training(X_XOR, y_XOR, 0.45, 100000)
prediction_XOR = nn.predict(X_XOR)
print('Acc :', nn.accurary(y_pred = nn.predict(X_XOR), y_true = y_XOR))


# In[11]:


def show_result(x, y, y_pred):
    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize = 18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    
    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize = 18)
    for i in range(x.shape[0]):
        if y_pred[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    
    plt.show()


# In[12]:


show_result(X_linear, y_linear, prediction_linear)
show_result(X_XOR, y_XOR, nn.predict(X_XOR))

