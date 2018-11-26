from src.DataManager_td import DataManager
from src.NeuralNetwork_td import NeuralNetwork
import numpy as np

def main():
    a = DataManager()
    a.loadData()
    b = NeuralNetwork()
    b.createModel()
    
    b.train(a.train_data, a.train_labels, a.eval_data, a.eval_labels, epochs = 100)
    b.saveModel("test.h5")

    # b.loadModel()
    # print("Evaluate : {}".format(b.evaluate(a.eval_data, a.eval_labels)))

if __name__ == "__main__":
    main()
