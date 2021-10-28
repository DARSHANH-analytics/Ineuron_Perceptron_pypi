import pandas as pd
import numpy as np
from tqdm import tqdm
import logging as logging

class perceptron:
    def __init__(self,lr,epochs):
        self.lr = lr
        self.epochs = epochs
                
    def fit(self,x,y):
        self.x = x
        self.y = y
        logging.info('------inputs---------')
        logging.info(self.x)
        #x_with_bias = np.concatenate([self.x,-np.ones((self.x.shape[0],1))],axis=1)
        #x_with_bias = pd.concat([self.x,pd.Series(-np.ones(self.x.shape[0]))],axis=1)
        x_with_bias = pd.concat([self.x,pd.Series(-np.ones(len(self.x)))],axis=1)
        logging.info('------x_with_bias---------')  
        logging.info(x_with_bias)
        self.weights = np.random.randn(x_with_bias.shape[1])
        logging.info('------weights---------')
        logging.info(self.weights)
        for e in tqdm(range(self.epochs), total=self.epochs, desc="training the model"):
            z = np.dot(x_with_bias,self.weights)
            y_pred = np.where(z>0,1,0)
            logging.info('------y_pred---------')   
            logging.info(y_pred)
            self.y_error = self.y - y_pred
            logging.info('------y_error---------')
            logging.info(self.y_error)
            logging.info('------x_with_bias.T---------Transforming to match the error matrix since column of first matrix to match \
the row of second matrix')
            if min(self.y_error) == 0 and max(self.y_error) == 0 :
                break;
            logging.info(x_with_bias.T)            
            self.weights = self.weights + self.lr * np.dot(x_with_bias.T,self.y_error)
            logging.info(f"updated weights after epoch:\n{e} : \n{self.weights}")            
    
    def predict(self,x):
        x_with_bias = np.concatenate([x,-np.ones((len(x),1))],axis=1)
        z = np.dot(x_with_bias,self.weights)
        return np.where(z>0,1,0)
    
    def total_loss(self):
        total_loss = np.sum(self.y_error)
        logging.info(f"total loss: {total_loss}")
        return total_loss