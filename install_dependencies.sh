# Coverage
sudo apt-get install -y -qq lcov

# OpenCV install
# sudo apt-get update -y
sudo apt-get install -y build-essential
sudo apt-get install -y cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install -y python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
# sudo apt-get install -y libopencv-dev
# sudo apt-get install -y libopencv-contrib-dev
# export CC="gcc $(pkg-config --cflags --libs opencv)"


# # Download v3.3.0
# curl -sL https://github.com/Itseez/opencv/archive/3.3.0.zip > opencv.zip
# unzip opencv.zip
# cd opencv-3.3.0
# mkdir build
# cd build
# cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_V4L=ON -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=ON -D WITH_QT=ON -D WITH_OPENGL=ON ..
# make -j4
# sudo make install
sudo apt-get install libopencv-dev

sudo sh -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf'
sudo ldconfig
cd ../../
