import numpy as np 
import pandas as pd

import tensorflow as tf



class LSTMRegressionModel(tf.keras.Model):
    def __init__(self, dropout_rate: float, hidden_dim_1: int, hidden_dim_2: int, output_dim: int):
        super(LSTMRegressionModel, self).__init__()
        self.lstm_1 = tf.keras.layers.LSTM(hidden_dim_1, return_sequences=True)
        self.layers_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.lstm_2 = tf.keras.layers.LSTM(hidden_dim_2, return_sequences=False)
        self.layers_dropout_2 = tf.keras.layers.Dropout(dropout_rate)
        self.dense = tf.keras.layers.Dense(output_dim)
        
    def call(self, inputs, training=False):
        

class ListNetLoss(tf.keras.losses.Loss): 
    def __init__(self, name="listnet_loss"):
        super().__init__(name=name)
        
    def call(self, y_true, y_pred): 
        p_true = tf.nn.softmax(y_true)
        p_pred = tf.nn.softmax(y_pred)
        loss = -tf.reduce_sum(p_true * tf.math.log(p_pred + 1e-10), axis=-1)
        return tf.reduce_mean(loss)
        
        