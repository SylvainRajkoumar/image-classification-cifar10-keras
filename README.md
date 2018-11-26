# image-classification-cifar10-keras
Initiation project on artificial intelligence applied to computer vision.
The goal is to translate the structure of a neural network initially implemented under cuda-convnet for use under Keras.
We achieved ~80% accuracy. 
![loss_100epochs_1dropout](https://user-images.githubusercontent.com/26735996/49030751-800dfc00-f1a8-11e8-821c-89e43699b90e.png)
![accuracy_100epochs_1dropout](https://user-images.githubusercontent.com/26735996/49030761-856b4680-f1a8-11e8-8297-29366e76a033.png)


## Prerequisites
Python 3.7 version 64-Bit Graphical Installer https://www.anaconda.com/download/

## Installing

### Python requirements
Once Anaconda is installed you can download the requirements for this project using the Anaconda Navigator  

Example for tensorflow-gpu :   

![capture](https://user-images.githubusercontent.com/26735996/48300550-0c010200-e4e0-11e8-8682-6cd7cf017bd1.PNG)

You'll need to do the same for the following packages :   
```
numpy  
opencv  
matplotlib  
tensorflow  
keras  
```

If you have an Nvidia gpu which support CUDA :  
```
tensorflow-gpu  
keras-gpu  
```

## Running
```
python main.py
```

