import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

class DataManager(object):

    def __init__(self):
        self.train_data = None
        self.train_labels = None
        self.eval_data = None
        self.eval_labels = None

    
    def loadData(self):
        print("Chargement du dataset CIFAR 10 ...")
        (self.train_data, self.train_labels), (self.eval_data, self.eval_labels) = cifar10.load_data()
        self.train_data = self.train_data / 255.0
        self.eval_data =  self.eval_data / 255.0
        print("Chargement terminé")
 

if __name__ == '__main__':
    a = DataManager()
    a.loadData()