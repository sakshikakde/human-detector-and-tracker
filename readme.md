# Human Detector and Tracker
hi
Module to detect and track humans and return their coordinate position.

[![Build Status](https://app.travis-ci.com/sakshikakde/human-detector-and-tracker.svg?branch=main)](https://app.travis-ci.com/github/sakshikakde/human-detector-and-tracker)
[![Coverage Status](https://coveralls.io/repos/github/sakshikakde/human-detector-and-tracker/badge.png?branch=main)](https://coveralls.io/github/anubhavparas/human-detector-and-tracker?branch=main)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
---

Authors: 
- [Anubhav Paras](https://github.com/anubhavparas) (Driver)
- [Sakshi Kakde](https://github.com/sakshikakde) (Navigator)
- [Siddharth Telang](https://github.com/siddharthtelang) (Design Keeper)

## Overview

Object detection has become an essential part of all robotic applications. A robot perceives the environment through its sensors and on the basis of the data collected, it performs some action. Specifically, human detection is a core problem in object detection. Avoiding collision with humans during the work process is a critical task for any developer who is designing the robotic application. This avoidance is strictly enforced to avoid any harm, damage or loss of (in the worst case scenario) human life. To add to this, object detection and particularly human detection in a dynamic environment is a very challenging task and demands high precision.

Numerous human object detectors have been developed till date which work very smoothly and accurately and their applications are growing rapidly. In addition it is one of the most popular research topics. With the advent of machine learning it has become easier to design these trackers given a large data set. However, it takes a long time to train such models and require high computational power and efficiency.

Acme robotics would be launching their robotics based product next year and a human detection and tracking system is the heart of that product. They have given us the complete ownership of designing and developing this detector and tracker. For this application, we aim to use Histogram of Features(HoG) with a Support Vector Machine(SVM) classifier.

Initially, we aim to train our model to extract features which would recognize humans in any given data set. We plan to use the INRIA Person Dataset to train our model. Given the data, a HoG would be generated. Next, these HOG vectors or features will be used to train a SVM classifier to detect the location of a human in any image. An error function will calculate the error made in classification which would be then fed to the classifier as a feedback to improve the accuracy. After the training is completed, the classifier would be able to detect the human position and draw a bounding box around it in real time feed from Monocular camera. In addition to this, the output of the classifier would be given to a tracking system to keep track of human(s) in real-time. Moreover, the tracked output will also be converted from image and camera frame to robot’s frame for further processing.

## Licence
MIT License

Copyright (c) 2021 Anubhav Paras, Sakshi Kakde, Siddharth Telang
```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## [QuadChart](./docs/QuadChart_Human_Detector_and_Tracker.pdf)

## [Proposal Report](./docs/Human_Detector_And_Tracker_Proposal.pdf)
   [Proposal video](https://youtu.be/RNsnlIrbCxQ)

## Deliverables
- Project: Human(s) obstacle detector and tracker - Output in robot reference frame.
- Overview of proposed work including timeline, risks, and mitigations.
- UML diagrams.
- Github repository with [README](./readme.md)
- Travis code coverage setup with Coveralls.
- Developer-level documentation.

## Potential Risks and Mitigation
- HoG is not scale invariant.
- Missed detections: It may happen that the model sometimes misses the detection of human presence. In such a case, deploying more than one and different models can be more useful.
- False and duplicate detection: Can be reduced by training the model such that the parameters are well tuned and the loss on validation data set is minimum.
- Unreliable detection boundary: Can be made accurate by selecting an optimum classifier, like the Bayes classifier.
- Flickers in detection: Filter can be deployed for a better estimate.

## Design
The following are the proposed designs for the human detector and tracker: 

![alt text](./UML/initial/HumanDetectorTracker_ClassDiagram.png?raw=true "Human Detecttor Class Diagram")

Activity flow diagrams can be found [here](./UML/initial/).

## Development process
Agile Development Process will be used in the development process with Test-Driven Development.

## [Product Backlog](https://docs.google.com/spreadsheets/d/1DSndkYyvKeA0DBJtzWZuSUIHXLhYmBShil_r_3koDa0/edit#gid=0)

## [AIP Sprint Sheet](https://drive.google.com/drive/folders/1sIxOGZRi_CUdVObPzQek-mL8ET90F8fe?usp=sharing)

## Tools and Technologies used
- Ubuntu 18.04(LTS)
- C++ 14+
- CMake
- OpenCV
- Travis CI
- Coveralls

## Dependencies with licenses
- OpenCV 4.5.0 (covered under the open-source Apache 2 License)
- Eigen 3.4 the Mozilla Public License 2.0
- Boost 1.65 Boost software license
- GMock BSD 3-Clause "New" or "Revised" License
- GTest BSD 3-Clause "New" or "Revised" License

## Dataset

We are using the INRIA dataset.

Instructions to fetch the dataset:
- Run
```
wget ftp://ftp.inrialpes.fr/pub/lear/douze/data/INRIAPerson.tar
```
- Extract the images from the dataset into a [testdata](./data/testdata) folder of the repository.



## Standard install via command-line

### Install Dependencies
```
# Coverage
sudo apt-get install -y -qq lcov

# OpenCV install
sudo apt-get install -y build-essential
sudo apt-get install -y cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install -y python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev

# Download v3.3.0
curl -sL https://github.com/Itseez/opencv/archive/3.3.0.zip > opencv.zip
unzip opencv.zip
cd opencv-3.3.0
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_V4L=ON -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=ON -D WITH_QT=ON -D WITH_OPENGL=ON ..
make -j4
sudo make install

sudo sh -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf'
sudo ldconfig
cd ../../
```

### Running the code

```
git clone --recursive https://github.com/anubhavparas/human-detector-and-tracker.git
cd <path to repository>
mkdir build
cd build
cmake ..
make
Run tests: ./test/cpp-test
Run program: ./app/shell-app
```
or run: 
```
sh build_coverage_off.sh
```

## Building for code coverage
Install code-coverage tool, else the code coverage command will not work. It is a one time installation: 
```
sudo apt-get install lcov
```
```
cmake -D COVERAGE=ON -D CMAKE_BUILD_TYPE=Debug ../
make
make code_coverage
```

or run: 
```
sh build_coverage_on.sh
```

This generates a index.html page in the build/coverage sub-directory that can be viewed locally in a web browser.

## Run cppcheck and cpplint
Run cppcheck: Results are stored in `./results/cppcheck_process.txt`, `./results/cppcheck_result.txt` 
```
sh run_cppcheck.sh
```

Run cpplint: Results are stored in `./results/cpplint_result.txt`
```
sh run_cpplint.sh
```



