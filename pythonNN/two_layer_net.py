import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient

import pickle
import csv
import numpy as np

class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
        self.params ={}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self,x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(x, W1) + b1

        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y

    def loss(self,x,t):
        y=self.predict(x)

        return cross_entropy_error(y,t)

    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy


    def numerical_gradient(self,x,t):
        loss_W=lambda W:self.loss(x,t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads


#numerical_gradientの高速版
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads

    def save_params(self,file_name="params.pkl"):
        params={}
        for key,val in self.params.items():
            params[key] = val#?
        with open(file_name,'wb') as f:
            pickle.dump(params,f)

        #csvへの書き込み
        with open('c:\\GNnetwork_params\\W1.CSV','w',newline='') as f1:
            writer = csv.writer(f1)
            writer.writerows(self.params['W1'])

        with open('c:\\GNnetwork_params\\W2.CSV','w',newline='') as f2:
            writer = csv.writer(f2)
            writer.writerows(self.params['W2'])

        with open('c:\\GNnetwork_params\\b1.CSV','w',newline='') as f3:
            writer = csv.writer(f3)
            writer.writerow(self.params['b1'])

        with open('c:\\GNnetwork_params\\b2.CSV','w',newline='') as f4:
            writer = csv.writer(f4)
            writer.writerow(self.params['b2'])

    def load_params(self,file_name="params.pkl"):
        with open(file_name,'rb') as f:
            params = pickle.load(f)
        for key,val in params.items():
            self.params[key] = val