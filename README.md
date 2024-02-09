# SightAi_model

### A COMPLETE STEP BY STEP PROCEDURE:

#### A Detailed Overview of the System

- Technical Setup

![image](https://github.com/KRISH2832/SightAi_model/assets/143683619/473fe578-118e-4cf0-a45c-4aafbebadcfc)


The system is set up in such a way where an android application(assuming you are implementing it on a android device) will capture real-time frames and will send it to a laptop based Networked Server where all the computations take place.
The Laptop Based Server will be using a pre-trained SSD detection model trained on <b>COCO DATASETS</b>. It will then test and the output class will get detected with an accuracy metrics.
After testing with the help of voice modules the class of the object will be converted into a default voice notes which will then be sent to the blind victims for their assistance.
Along with the object detection , we have used an alert system where approximate will get calculated. If that Blind Person is very close to the frame or is far away at a safer place , it will generate voice-based outputs along with distance units.
A very notable point is it uses Resnet based architectural approach for feature extraction.

![image](https://github.com/KRISH2832/SightAi_model/assets/143683619/f8e3835a-170c-423d-b05f-378b663bd4a0)

#### IMPLEMENTATION
Efficient Implementation of this Model depends upon the compatibility with python and library installation hurdles. To be honest , this was among one of the most challenging phase which I felt in building this project. Thankful to STACKOVERFLOW and Python Unoffiicial Binary Releases for having pre-builded files uploaded and you can just download it from here as per your system’s compatibility.

- TENSORFLOW APIs

The TensorFlow object detection API is basically a structure build for creating a deep learning network that solves the problems for object detection.There are trained models in their framework and they refer it as Model Zoo .This includes a collection of COCO dataset, the KITTI dataset, and the Open Images Dataset.
Here, we are primarily focused on COCO DATASETS.

TensorFlow Object Detection API depends on the following libraries:

1. Protobuf 3.0.0
2. Python-tk
3. Pillow 1.0
4. lxml
5. tf-slim
6. slim
7. Jupyter notebook
8. Matplotlib
9. Tensorflow (1.15.0)
10. Cython
11. contextlib2
12. cocoapi

#### GETTING READY WITH THE SYSTEM

1. TensorFlow Installation
A typical user can simply install it on Anaconda Prompt with following commands.
```
pip install tensorflow
pip install tensorflow-gpu
```



2. Now you need to download the TensorFlow model repository from
```
git clone https://github.com/tensorflow/models.git
```

3.PROTOBUF COMPILATION

Next you have to Convert the .protos file to .py file extensions.

This can be done via Protobuf Compilation. For achieving this we we’ll need Google Protobuf Releases. Head on to the link and Download the protobuf version which satisfies your system compatibility. I prefer win_64 which supports my system. After downloading and extracting it make sure to add it’s path in the environment variable otherwise it’ll still give you errors as ‘proto is not recognised as an internal or external batch file’. So,make sure it is done!

Further , inside of tensorflow/models/research/ directory :

Hit the following command:
```
protoc object_detection/protos/.proto --python_out=.
```

You’ve successfully converted your protos file into python files.



#### MODELS
Now, a bunch of pre-trained models are with Tensorflow . You can use any one of them. They are pretty good and depending upon your system specifications you can choose one. For a faster accuracy you can go with SSD DETECTION and for better accuracy you can go with MASK RCNN but most of the system shows smooth performance with SSD Mobile_Net DETECTION . So, I’ll elaborate SSD ALGORITHM. You can check other models here:

- SSD ARCHITECTURE


THE SSD ARCHITECTURE
SSD has two components: SSD head and a backbone model.

Backbone model basically is a trained image classification network as a feature extractor. Like ResNet this is typically a network trained on ImageNet from which the final fully connected classification layer has been removed.

The SSD head is just one or more convolutional layers added to this backbone and the outputs are interpreted as the bounding boxes and classes of objects in the spatial location of the final layers activations.We are hence left with a deep neural network which is able to extract semantic meaning from the input image while preserving the spatial structure of the image albeit at a lower resolution.

For an input image ,the backbone results in a 256 7x7 feature maps in ResNet34 . SSD divides the image using a grid and have each grid cell be responsible for detecting objects in that region of the image. Detecting objects basically means predicting the class and location of an object within that region.

- Anchor box

Multiple anchor/prior boxes can be assigned to each grid cell in SSD. These assigned anchor boxes are pre-defined and each one is responsible for a size and shape within a grid cell.Matching phase is used by SSD while training, so that there’s an appropriate match to anchor box with the bounding boxes of each ground truth object within an image. For predicting that object’s class and its location the anchor box with the highest degree of overlap with an object is responsible.Once the network has been trained,this property is used for training the network and for predicting the detected objects and their locations. Practically, each anchor box is specified with an aspect ratio and a zoom level. Well,we know that all objects are not square in shape. Some are shorter ,some are longer and some are wider, by varying degrees. The SSD architecture allows pre-defined aspect ratios of the anchor boxes to account for this.The different aspect ratios can be specified using ratios parameter of the anchor boxes associated with each grid cell at each zoom/scale level.

It is not mandatory for the anchor boxes to have the same size as that of the grid cell.The user might be interested in finding both smaller or larger objects within a grid cell. In order to specify how much the anchor boxes need to be scaled up or down with respect to each grid cell ,the zooms parameter is used.

- MOBILENET

This model is based on the ideology of THE MobileNet model based on depthwise separable convolutions and it forms a factorized Convolutions.
These converts a basic standard convolutions into a depthwise convolutions.This 1 × 1 convolutions are also called as pointwise convolutions.
For MobileNets to work, these depthwise convolutions applies a general single filter based concept to each of the input channels.
These pointwise convolutions applies a 1 × 1 convolutions to merge with the outputs of the depthwise convolutions.
As a standard convolution both filters combines the inputs into a new set of outputs in one single step. The depthwise identifiable convolutions splits this into two layers — a separate layer for the filtering purpose and the other separate layer for the combining purpose. This factorization methodology has the effect of drastically reducing the computation and that of the model size.


Depth estimation or extraction feature is nothing but the techniques and algorithms which aims to obtain a representation of the spatial structure of a scene. In simpler words, it is used to calculate the distance between two objects. Our prototype is used to assist the blind people which aims to issue warning to the blind people about the hurdles coming on their way. In order to do this, we need to find that at how much distance the obstacle and person are located in any real time situation. After the object is detected rectangular box is generated around that object.


Distance Approximations
If that object occupies most of the frame then with respect to some constraints the approximate distance of the object from the particular person is calculated. Following code is used to recognize objects and to return the information of the distance and location.


Here, we have established a Tensorflow session comprised of Crucial Features for Detection. So, for further analysis iteration is done through the boxes. Boxes are an array, inside of an array. So, for iteration we need to define the following conditions.


Index of box in boxes array is represented by i. Analysis of the score of the box is done by index. It is also used to access class. Now the width of the detected object is measured. This is done by asking the width of an object in terms of pixels.


We got the center of two by subtracting the same axis start coordinates and dividing them by two. In this way the centre of our detected rectangle is calculated. And at the last, a dot is drawn in the centre. The default parameter for drawing boxes is a score of 0.5. if scores[0][i] >= 0.5 (i.e. equal or more than 50 percent) then we assume that the object is detected. if scores[0][i] >= 0.5:


In the above formula, mid_x is centre of X axis and mid_y is centre of y axis. If the distance apx_distance < 0.5 and if mid_x > 0.3 and mid_x < 0.7 then it can be concluded that the object is too close from the particular person. With this code, relative distance of the object from a particular person can be calculated. After the detection of object the code is used to determine the relative distance of the object from the person. If the object is too close then signal or a warning is issued to the person through voice generation module.

- VOICE GENERATION MODULE

After the detection of an object, it is utmost important to acknowledge the person about the presence of that object on his/her way. For the voice generation module PYTTSX3 plays an important role.
Pyttsx3 is a conversion library in Python which converts text into speech.
This library works well with both Python 2 and 3.
To get reference to a pyttsx. Engine instance, a factory function called as pyttsx.init() is invoked by an application.
Pyttsx3 is a tool which converts text to speech easily.

This algorithm works as whenever an object is being detected, approximate distance is being calculated,with the help of cv2 library and cv2.putText() function, the texts are getting displayed on to the screen. To identify the hidden text in an image,we use Python-tesseract for character recognition.
OCR detects the text content on images and encodes it in the form which is easily understood by the computer.
This text detection is done by scanning and analysis of the image.
Thus, the text embedded in images are recognized and “read” using Python-tesseract. Further these texts are pointed to a pyttsx.Engine instance, a factory function called as pyttsx.init() is invoked by an application. During construction, a pyttsx.driver.DriverProxy object is initialized by engine which is responsible for loading a speech engine driver from the pyttsx.drivers module. After construction, an object created by an engine is used by the application to register and unregister event callbacks; produce and stop speech; get and set speech engine properties; and start and stop event loops.


Audio commands are generated as output. If the object is too close then it states “Warning: The object (class of object) is very close to you. Stay alert!”. Else if the object is at a safer distance then then a voice is generated which says that “The object is at safer distance”. This is achieved with the help of certain libraries like pytorch, pyttsx3, pytesseract and engine.io .

- Pytorch is primarily a machine learning library.
Pytorch is mainly applied to the audio domain.
Pytorch helps in loading the voice file in standard mp3 format. It also regulates the rate of audio dimension. Thus, it is used to manipulate the properties of sound like frequency, wavelength and waveform. The numerous availability of options for audio synthesis can also be verified by taking a look at the functions of Pytorch.

#### TESTING
Third Party App provides ease and freedom in the field of app development. It brings efficiency and also helps in fast delivery of the output. Third Party App allows you to divide your work in parts and helps you to focus on the core part of app or any system. This strategy helps in the development of good and quality software. We can pass on the Features of the Third Party App to the system.

1. At first, we are capturing real time images from the rear camera[4] of the mobile handset of blind people and a connection is established between mobile phone and system in laptop and then those images are sent from the mobile phone to laptop.

2. This connection is done by a Third party app which is installed in the mobile phone of the person. All the real time images which get captured by the rear camera of the mobile phone are first transferred to the Third party app in the mobile phone and then those images are sent in laptop where they are processed for some further conclusions.

3. The system in laptop will test it using its APIs and SSD ALGORITHM and it detects the confidence accuracy of the image which it is testing. We reached 98% accuracy for certain classes like books, cups, remote.

4. After testing the images we are generating an output on the laptop based system and its prediction is being translated into voice with voice modules and sent to the blind person with the help of wireless audio support tools.




Output
Below are the objects on which it was tested and it gave the following result which were analysed further with the help matplotlib libraries.

![image](https://github.com/KRISH2832/SightAi_model/assets/143683619/3c08ab56-b823-4110-8667-2943d63ba3ad)

THE ACCURACY OF CUP IS 99%


![image](https://github.com/KRISH2832/SightAi_model/assets/143683619/20377763-3cc1-4c3e-a5ed-544e00d58880)

THE FINAL ACCURACY OF REMOTE IS 98%

![image](https://github.com/KRISH2832/SightAi_model/assets/143683619/13cc5210-b902-4f7e-889b-aaea8581b6e0)

THE FINAL ACCURACY OF BED IS 98%


![image](https://github.com/KRISH2832/SightAi_model/assets/143683619/58385055-783f-46f4-b001-b785463f31af)

THE FINAL ACCURACY OF CHAIR IS 96%


![image](https://github.com/KRISH2832/SightAi_model/assets/143683619/38e7057b-6599-46ae-9180-f8b6069ce3e4)

THE FINAL ACCURACY OF TV IS 96%


