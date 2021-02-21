# Sign Language Recognition

* Real Time application to detect sign of hand and translate to letters. Try to help deaf people can commute with other people.
* Includes pre-trained model by Yolov4, and program predicts sign letter by getting video via webcame.
* Built with Python, Keras+Tensorflow and OpenCV (for video capturing and manipulation).
* Implement the system consists of steps: prepare data, train model, pre-process video, predict sign letter.

## Getting started
### Requirements
* python  >= 3.6
* opencv-python >= 3.4.1.15
* keras >= 2.2.0
### Running 
* Run ```final.py ``` in folder source
* Press B to confirm background
* Give hand and perform your hand in square frame via webcame and then system will show the character you are performing. 
### Implement
#### Prepare data
My friend took the pictures and labeled all of  .. images by himself. And he shares the dataset for this project. The dataset is divided into 25 classes.

![alt text](https://github.com/naot97/sign_language_recognition/blob/master/source/Characters.bmp?raw=true)

#### Train model

#### Pre-process video
Because input of the model is an image, so the program needs an image to predict the letter. We know a video is a series of frames, whenever the counter is divisible by 50, the program get a frame to predict. And due to the frame's area is too large. the program only takes the marked region (the green rectangle in Image) to predict.

The Background subtraction technique is applied to remove background of frame. Background subtraction (BS) is a common and widely used technique for generating a foreground mask (namely, a binary image containing the pixels belonging to moving objects in the scene) by using static cameras. As the name suggests, BS calculates the foreground mask performing a subtraction between the current frame and a background model, containing the static part of the scene or, more in general, everything that can be considered as background given the characteristics of the observed scene.
 




