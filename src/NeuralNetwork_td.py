import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork(object):

    def __init__(self):
        self.model = None

    def createModel(self):
        print("Création du modèle ...")
        self.model = keras.Sequential()

        self.model.add(keras.layers.Conv2D(filters=32, kernel_size=5, strides=1, padding="same",
                                            input_shape=(32, 32, 3), activation=tf.nn.relu, data_format='channels_last')) #CONV1
        self.model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")) #POOL1
        self.model.add(keras.layers.BatchNormalization()) #RNORM1
        self.model.add(keras.layers.Conv2D(filters=32, kernel_size=5, strides=1, padding="same", activation=tf.nn.relu)) #CONV2
        self.model.add(keras.layers.AveragePooling2D(pool_size=3, strides=2)) #POOL2
        self.model.add(keras.layers.BatchNormalization()) #RNORM2
        self.model.add(keras.layers.Conv2D(filters=64, kernel_size=5, strides=1, padding="same", activation=tf.nn.relu)) #CONV3
        self.model.add(keras.layers.AveragePooling2D(pool_size=3, strides=2)) #POOL3
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(10, activation = tf.nn.softmax)) #FC10

        self.model.compile(optimizer=keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

        print("Création du modèle réussite")

    def train(self, train_data, train_labels, eval_data, eval_labels, epochs):
        """Train the keras model
        
        Arguments:
            train_data {np.array} -- The training image data
            train_labels {np.array} -- The training labels
            epochs {int} -- The number of epochs to train for
        """
        
        # datagen = keras.preprocessing.image.ImageDataGenerator(
        #     featurewise_center=True,
        #     featurewise_std_normalization=True,
        #     rotation_range=20,
        #     width_shift_range=0.2,
        #     height_shift_range=0.2,
        #     horizontal_flip=True)

        # datagen.fit(train_data)
        # self.model.fit_generator(datagen.flow(train_data, train_labels, batch_size=32),
        #             steps_per_epoch=len(train_data) / 32, epochs=epochs)

        # for e in range(epochs):
        #     batches = 0
        #     for x_batch, y_batch in datagen.flow(train_data, train_labels, batch_size=32):
        #         self.model.fit(x_batch, y_batch)
        #         print(self.evaluate(eval_data, eval_labels))
        #         batches += 1
        #         if batches >= len(train_data) / 32:
        #             # we need to break the loop by hand because
        #             # the generator loops indefinitely
        #             break

        print("Entrainement du réseau de neurones en cours ... ")
        history = self.model.fit(train_data, train_labels, epochs = epochs, batch_size=128, validation_split=0.11)
        print("Entrainement terminé")

        print(history.history.keys())

        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


    def evaluate(self, eval_data, eval_labels):
        """Calculate the accuracy of the model
        
        Arguments:
            eval_data {np.array} -- The evaluation images
            eval_labels {np.array} -- The labels for the evaluation images
        """
        return self.model.evaluate(eval_data, eval_labels)[1]

    def test(self, test_data):
        """Make predictions for a list of images and display the results
        
        Arguments:
            test_data {np.array} -- The test images
        """
        predictions = self.model.predict(test_data)
        return predictions

    ## Exercise 7 Save and load a model using the keras.models API
    def saveModel(self, saveFile="model.h5"):
        """Save a model using the keras.models API
        
        Keyword Arguments:
            saveFile {str} -- The name of the model file (default: {"model.h5"})
        """
        self.model.save(saveFile)
        print("Le modèle a bien été sauvegardé sous le nom {}".format(saveFile))

    def loadModel(self, saveFile="model.h5"):
        """Load a model using the keras.models API
        
        Keyword Arguments:
            saveFile {str} -- The name of the model file (default: {"model.h5"})
        """
        self.model = keras.models.load_model(saveFile)
        print("Le modèle a bien été chargé depuis le fichier {}".format(saveFile))

if __name__ == '__main__':
    a = NeuralNetwork()
    a.createModel()