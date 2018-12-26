# movidius-rpi

## Real-time face detection example with OpenVINO running on Raspberry Pi

### Requirements
* Raspberry Pi 3 Model B
* Movidius Neural Compute Stick 2 (not tested yet on NCS)
* Webcam
* Display (Optional)

### Installation

To install the Intel® Distribution of OpenVINO™ Toolkit for Raspbian, follow [these instructions](https://software.intel.com/articles/OpenVINO-Install-RaspberryPI)

### Running the app

Command example:

python realtime_facedetection.py --display 1

In order to test and debug the app, I have added the video and ran the app on VM. In order to process the video instead of webcam stream, run the following:

--input videos/dive.mp4

### Credits

Thanks to [Adrian Rosenbrock](https://github.com/jrosebr1) for the following helpful tutorials:

* [Real-time object detection on the Raspberry Pi with the Movidius NCS](https://www.pyimagesearch.com/2018/02/19/real-time-object-detection-on-the-raspberry-pi-with-the-movidius-ncs/)
* [OpenCV People Counter](https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/)
