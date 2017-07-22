![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

#Darknet#
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).


Discussion
We would like to use DGX and train our models under evaluation and check their performance.
We had a round of evaluations mainly with pre-trained weights, YOLO and Faster-RCNN.

As we need real-time performance, we are going ahead with YOLO.
We are using the neural network framework: “darknet“ to explore the models, especially YOLO here.
Why Darknet?
1) The framework is written in C and CUDA.
2) Code is easy to understand.
3) We can easily build the code, tweak the framework for our evaluation needs and application prototyping (especially for AI City Challenge Track 2).
https://pjreddie.com/darknet/
NVIDIA DGX
Index:
ssh aic@10.31.229.235

Darknet Dependencies
OpenCV, gstreamer-1.0, gtk-3.0-dev.
Get Darknet
git clone https://github.com/pjreddie/darknet.git
vim Makefile 
> and change to:
GPU=1
OPENCV=1
CUDANN=1
$make

Installing Dependencies
Check References below (if you’re installing this on personal machines).
In DGX, NVIDIA has provided us an image: “aic/darknet” which might be a fork of https://hub.docker.com/r/jh800222/darknet/.
You can extend the image by:
1) nvidia-docker run -it -v /datasets:/datasets -v /data:/data -v /home/aic:/home/aic aic/darknet
2) exit
3) nvidia-docker ps -a
[Not down the container-ID]
4) nvidia-docker commit ac9b2bceb2b7 yournewcontainername/new_darknet_image_name
Say, example:
nvidia-docker commit ac9b2bceb2b7 test/darknet_test
4) Run your new image: 
nvidia-docker run -it -v /datasets:/datasets -v /data:/data -v /home/aic:/home/aic test/darknet_test
5) Remember to commit back your work using steps 2) and 3) above.

Training YOLO (or any model) using darknet
Generating dataset
NVIDIA has provided us the dataset in darknet format at /dataset/aic*-darknet/.
For more info on generating this format, 
Dataset folders (generating train.txt and valid.txt)
Training dataset information (train.txt)
Generate train.txt with details of training data images (path to all jpegs at dataset/train/images/).
To do that: (you can train only 1080p images or choose to train all data)
$cd /datasets/aic1080-darknet/train/images/
$find `pwd` > /workspace/darknet/train.txt
$vi /workspace/darknet/train1080.txt
And delete the first line “/datasets/aic*-darknet/train/images”

Advanced (if you want to include lower resolution data as well) - Team 1 is experimenting on this, don’t do this now.
$cd /datasets/aic540-darknet/train/images/
$find `pwd` > /workspace/darknet/train540.txt
[delete line one in the text file - just the way we did for train.txt above]
$cd /datasets/aic480-darknet/train/images/
$find `pwd` > /workspace/darknet/train480.txt
[delete line one in the text file - just the way we did for train.txt above]
$cd /workspace/darknet
$cat train540.txt >> train.txt
$cat train480.txt >> train.txt

Validation dataset information (valid.txt)
Generate valid.txt with details of validation data images (path to all jpegs at dataset/val/images/).
To do that: (you can train only 1080p images or choose to train all data)
$cd /datasets/aic1080-darknet/val/images/
$find `pwd` > /workspace/darknet/valid.txt
$vi /workspace/darknet/valid.txt
And delete the first line “/datasets/aic*-darknet/val/images”

Advanced (where we wish to train 540p and 480p images):
Generate valid*.txt in a way similar to the one followed for train*.txt we followed above.
$cat valid540.txt >> valid.txt
$cat valid480.txt >> valid.txt

Now we have train.txt and valid.txt files which are essential input to yolo.c file below which read these and loads images for training.

[Ignore this] Create 2 folders in the darknet/ folder:
images : all the jpeg images copied from: /datasets/aic1080-darknet/train/images
labels: all txt files copied from /datasets/aic1080-darknet/train/annotations
Code changes
Quick Hack: [Don’t read below and use my darknet code at: “”]
examples/yolo.c has the code to train and validate YOLO’s neural network as defined in a config file, say cfg/yolo-aic.cfg.
Changes required to start training:
Change the class labels 
Paths to train.txt, valid.txt.
“jpeg” support:
In file examples/yolo.c and src/data.c 
Find the source line: “find_replace(labelpath, ".jpg", ".txt", labelpath);”
Just add below this line:
find_replace(labelpath, ".jpeg", ".txt", labelpath);
Change in files examples/yolo.c and src/data.c:
“labels” to “annotations” as our labels, the .txt files in darknet format are in a folder named “annotations”.

Configuring the YOLO model for your own classes
yolo-aic.cfg file (The Neural Network!)
Copy cfg/yolo-voc.2.0.cfg to cfg/yolo-aic.cfg and:
change line batch to batch=64
change line subdivisions to subdivisions=8
change line classes=20 to your number of objects
change line #237 from filters=125 to filters=(classes + 5)*5 (generally this depends on the num and coords, i.e. equal to (classes + coords + 1)*num)
For example, for 2 objects, your file yolo-aic.cfg should differ from yolo-voc.2.0.cfg in such lines:
[convolutional]
filters=35

[region]
classes=2
Also,  there are a number of factors determining each of the hyperparameters that we could play with. 
Read the paper: “https://arxiv.org/pdf/1506.02640.pdf” for a detailed idea.
Advanced (when training with multiple resolutions):
Change in the config file:
random=1

aic.names
Create file aic.names in the directory darknet/data, with objects names - each in new line
aic.data
Create file aic.data in the directory darknet/cfg, containing (where classes = number of objects):
classes= 2
train  = train.txt
valid  = valid.txt
names = data/aic.names
backup = backup/


Start Training!!
./darknet detector train cfg/aic.data cfg/yolo-aic.cfg /data/team1_darknet/darknet19_448.conv.23



Improving Object detection
Before training:
set flag random=1 in your .cfg-file - it will increase precision by training Yolo for different resolutions.
desirable that your training dataset include images with objects at different: scales, rotations, lightings, from different sides.
After training - for detection:
Increase network-resolution by set in your .cfg-file (height=608 and width=608) or (height=832 and width=832) or (any value multiple of 32) - this increases the precision and makes it possible to detect small objects:
you do not need to train the network again, just use .weights-file already trained for 416x416 resolution.
if error Out of memory occurs then in .cfg-file you should increase subdivisions=16, 32 or 64.

References
1) To understand how to provide dataset in darknet format (this part you can skip as AIC provided dataset in darknet format already - check /dataset/aic*-darknet/) and start training.
http://guanghan.info/blog/en/my-works/train-yolo/

2) FAQ, check:
https://groups.google.com/forum/#!forum/darknet

3) For more details on training:
https://github.com/AlexeyAB/darknet
This repo support darknet on windows as well - if you'd like to explore.

4) Further, and precise info on training can be understood here:
https://pjreddie.com/darknet/yolo/

5) Read in full this paper:
https://arxiv.org/pdf/1506.02640.pdf

