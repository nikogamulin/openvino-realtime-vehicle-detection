# movidius-rpi

## Vehicle detection examples with OpenVINO running on Raspberry Pi 3 Model B (RPi) and Intel® Core™ i7-4790K CPU @ 4.00GHz × 8 (desktop)

[![Car Detection with Intel NCS2 and Raspberry Pi](https://img.youtube.com/vi/HmZiPxM4OMk/0.jpg)](https://www.youtube.com/watch?v=HmZiPxM4OMk)

### Requirements
* Raspberry Pi 3 Model B
* Movidius Neural Compute Stick 2 (not tested yet on NCS)
* Webcam
* Display (Optional)

### Scripts
* realtime_vehicledetection.py (model: [vehicle-detection-adas-0002](https://github.com/opencv/open_model_zoo/blob/2018/intel_models/vehicle-detection-adas-0002/description/vehicle-detection-adas-0002.md)
* realtime_objectdetection.py (model: [mobilenet-ssd](https://github.com/opencv/open_model_zoo/blob/2018/model_downloader/README.md))
* realtime_objectdetection_and_tracking.py (model: [mobilenet-ssd](https://github.com/opencv/open_model_zoo/blob/2018/model_downloader/README.md))

### Installation

To install the Intel® Distribution of OpenVINO™ Toolkit for Raspbian, follow [these instructions](https://software.intel.com/articles/OpenVINO-Install-RaspberryPI)

### Model conversion to IR

mobilenet-ssd from model zoo has been transformed to IR with model optimizer with following command:

python3 ~/intel/computer_vision_sdk/deployment_tools/model_optimizer/mo.py --input_model ~/workspace/open_model_zoo/model_downloader/object_detection/common/mobilenet-ssd/caffe/mobilenet-ssd.caffemodel  --input_proto ~/workspace/open_model_zoo/model_downloader/object_detection/common/mobilenet-ssd/caffe/mobilenet-ssd.prototxt --data_type FP16

Change path according to your installation.

### Model Performance

| task (model)                                                             | Device  | FPS   |
| ------------------------------------------------------------------------ |:-------:| -----:|
| Vehicle Detection (vehicle-detection-adas-0002)                          | desktop | 6.46  |
| Object Detection (mobilenet-ssd)                                         | desktop | 18.31 |
| Object Detection And Tracking with default configuration (mobilenet-ssd) | desktop | 30.27 |
| Vehicle Detection (vehicle-detection-adas-0002)                          | RPi     | -     |
| Object Detection (mobilenet-ssd)                                         | RPi     | 6.90  |
| Object Detection And Tracking with default configuration (mobilenet-ssd) | RPi     | 10.17 |

### Credits

Thanks to [Adrian Rosenbrock](https://github.com/jrosebr1) for the following helpful tutorials:

* [Real-time object detection on the Raspberry Pi with the Movidius NCS](https://www.pyimagesearch.com/2018/02/19/real-time-object-detection-on-the-raspberry-pi-with-the-movidius-ncs/)
* [OpenCV People Counter](https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/)
