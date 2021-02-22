# Sign Language Recognition

* Real Time application to detect sign of hand and translate to letters. Try to help deaf people can commute with other people.
* Includes pre-trained model by Keras, and program predicts sign letter by getting video via webcame.
* Built with Python, Keras+Tensorflow and OpenCV (for video capturing and manipulation).
* Implement the system consists of steps: prepare data, train model, pre-process video, running and experiment.

## Getting started
### Requirements
* python  >= 3.6
* opencv-python >= 3.4.1.15
* keras >= 2.2.0
### Implement
#### Prepare data
My friend took the pictures and labeled all of 45240 100x100x3 images by himself. And he shares the dataset for this project. The dataset for train includes 2 group: images taken by the webcam and images which I augment from them. What is about to augment data? Data augmentation is another common pre-processing technique involves augmenting the existing data-set with perturbed versions of the existing images. Scaling, rotations and other affine transformations are typical. This is done to expose the neural network to a wide variety of variations. This makes it less likely that the neural network recognizes unwanted characteristics in the data-set. In our project, we used rotation, shift, zoom to augment our data. The dataset is divided into 26 classes ( 25 letters + 1 empty class). 

![alt text](https://github.com/naot97/sign_language_recognition/blob/master/characters.jpg?raw=true "Sign Language")

I used 40000 images for  trainning, 4240 for validation and 1000 for testing. Below is statistic of training data:

![alt text](https://github.com/naot97/sign_language_recognition/blob/master/staticstic.png?raw=true "Staticstic quantity of training dataset")

#### Train model

The model I used to train is a simple CNN. You can find the training code is [here](https://colab.research.google.com/drive/1QTtw_thu_woWTSz7U2ECXsDtZwBbbV6_?usp=sharing). The CNN contains three components:
* Convolutional layers, which apply a specified number of convolution filters to the image. For each subregion, the layer performs a set of mathematical operations to produce a single value in the output feature map. Convolutional layers then typically apply a ReLU activation function to the output to introduce nonlinearities into the model.
* Pooling layers, which downsample the image data extracted by the convolutional layers to reduce the dimensionality of the feature map in order to decrease processing time.
* Dense (fully connected) layers, which perform classification on the features extracted by the convolutional layers and downsampled by the pooling layers. In a dense layer, every node in the layer is connected to every node in the preceding layer.

![alt text](https://github.com/naot97/sign_language_recognition/blob/master/train_model.PNG?raw=true "Model CNN for training")

During training, dropout and data augmentation are used as main approaches to reduce overfitting. This consists of zooming up to 0.15, rotations up to 25 degree, spatial translations up to 0.25. My model was trained by using Ndivia Tesla K80 provided Google Colab.I have set-up 50 epochs for our training model. The learning rate start with value 0.001 and decrease over time with 0.005 on decay rate. Below is the training chart of the model:

![alt text](https://github.com/naot97/sign_language_recognition/blob/master/trainning%20accuracy%20over%20epochs.png?raw=true "Training accuracy over epochs")

![alt text](https://github.com/naot97/sign_language_recognition/blob/master/trainning%20loss%20over%20epochs.png?raw=true "Training loss over epochs")

![alt text](https://github.com/naot97/sign_language_recognition/blob/master/validation%20accuracy%20over%20epochs.png?raw=true "Validation accuracy over epochs")

![alt text](https://github.com/naot97/sign_language_recognition/blob/master/validation%20loss%20over%20epochs.png?raw=true "Validation loss over epochs")


#### Pre-process video
The Background subtraction technique is applied to remove background of frame. Background subtraction (BS) is a common and widely used technique for generating a foreground mask (namely, a binary image containing the pixels belonging to moving objects in the scene) by using static cameras. As the name suggests, BS calculates the foreground mask performing a subtraction between the current frame and a background model, containing the static part of the scene or, more in general, everything that can be considered as background given the characteristics of the observed scene.

![alt text](https://github.com/naot97/sign_language_recognition/blob/master/SB.PNG?raw=true "Background subtraction")

Because input of the model is an image, so the program needs an image to predict the letter. We know a video is a series of frames, whenever the counter is divisible by 50, the program get a frame to predict. And due to the frame's area is too large. the program only takes the marked region (the blue square in Image) to predict.

![alt text](https://github.com/naot97/sign_language_recognition/blob/master/demo.PNG?raw=true "Demo")

#### Running  
* Run ```final.py ``` in folder source
* Press B to confirm background
* Give hand and perform your hand in square frame via webcame and then system will show the character you are performing. 

![alt text](https://github.com/naot97/sign_language_recognition/blob/master/result.png?raw=true "Result")

#### Experiment
The performance of the model on reallife performs (1000 times with each class). These performs are in perfect light conditions.

![alt text](https://github.com/naot97/sign_language_recognition/blob/master/performance.PNG?raw=true "Experiment")

When I deploy the model in real time, my model is not good at recognize character X, M and N. These failures may come from the training data or the augment processing or slightly overfitting in the model. For non-static sign language recognition, I haven’t still find a way to recognize the character. In the future, I would like to improve the accuracy of the program, simultaneously I will study CNN 3D to analyse exactly the context of non-static sign language. Furthermore, I think it will be interesting if I combine voices in this program.


