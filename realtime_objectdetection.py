# USAGE
# python realtime_objectdetection.py --input videos/izola_highway.mp4 --display 1

# import the necessary packages
import os
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import time
import cv2
import numpy as np

running_on_rpi = False

os_info = os.uname()
if os_info[4][:3] == 'arm':
    running_on_rpi = True


writer = None
W = None
H = None

# check if optimization is enabled
if not cv2.useOptimized():
    print("By default, OpenCV has not been optimized")
    cv2.setUseOptimized(True)


# initialize the list of class labels our network was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ("background", "aeroplane", "bicycle", "bird",
           "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor")

image_for_result = None


def predict(frame, net):

    # Prepare input blob and perform an inference
    blob = cv2.dnn.blobFromImage(frame, 0.007843, size=(300, 300), mean=(127.5, 127.5, 127.5), swapRB=False, crop=False)
    net.setInput(blob)
    out = net.forward()
    out = out.flatten()

    predictions = []

    for box_index in range(100):
        if out[box_index + 1] == 0.0:
            break
        base_index = box_index * 7
        if (not np.isfinite(out[base_index]) or
                not np.isfinite(out[base_index + 1]) or
                not np.isfinite(out[base_index + 2]) or
                not np.isfinite(out[base_index + 3]) or
                not np.isfinite(out[base_index + 4]) or
                not np.isfinite(out[base_index + 5]) or
                not np.isfinite(out[base_index + 6])):
            continue


        object_info_overlay = out[base_index:base_index + 7]

        base_index = 0
        class_id = int(object_info_overlay[base_index + 1])
        conf = object_info_overlay[base_index + 2]
        if (conf <= args["confidence"] or class_id != 7):
            continue

        box_left = object_info_overlay[base_index + 3]
        box_top = object_info_overlay[base_index + 4]
        box_right = object_info_overlay[base_index + 5]
        box_bottom = object_info_overlay[base_index + 6]

        prediction_to_append = [class_id, conf, ((box_left, box_top), (box_right, box_bottom))]
        predictions.append(prediction_to_append)

    return predictions


def resize(frame, width, height=None):
    h, w, _ = frame.shape
    if height is None:
        # keep ratio
        factor = width * 1.0 / w
        height = int(factor * h)
    frame_resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame_resized


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", default=.5,
                help="confidence threshold")
ap.add_argument("-d", "--display", type=int, default=0,
                help="switch to display image on screen")
ap.add_argument("-i", "--input", type=str,
                help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
                help="path to optional output video file")
ap.add_argument("-r", "--resize", type=str, default=None,
                help="resized frames dimensions, e.g. 320,240")
args = vars(ap.parse_args())

# Load the model
net = cv2.dnn.readNet('models/mobilenet-ssd/FP16/mobilenet-ssd.xml', 'models/mobilenet-ssd/FP16/mobilenet-ssd.bin')

# Specify target device
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

# if a video path was not supplied, grab a reference to the webcam
if not args.get("input", False):
    print("[INFO] starting video stream...")
    if args["resize"] is not None:
        w, h = [int(item) for item in args["resize"].split(",")]
        vs = VideoStream(src=0, resolution=(w, h)).start()
    else:
        vs = VideoStream(src=0).start()
    time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
    print("[INFO] opening video file...")
    vs = cv2.VideoCapture(args["input"])

time.sleep(1)
fps = FPS().start()

# loop over frames from the video file stream
while True:
    try:
        # grab the frame from the threaded video stream
        # make a copy of the frame and resize it for display/video purposes
        frame = vs.read()
        frame = frame[1] if args.get("input", False) else frame

        H, W, _ = frame.shape
        if args["display"] > 0 or args["output"] is not None:
            image_for_result = frame.copy()

        # if we are supposed to be writing a video to disk, initialize
        # the writer
        if args["output"] is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                     (frame.shape[1], frame.shape[0]), True)

        # use the NCS to acquire predictions
        predictions = predict(frame, net)

        # loop over our predictions
        for (i, pred) in enumerate(predictions):
            # extract prediction data for readability
            (label, pred_conf, pred_boxpts) = pred
            ((x_min, y_min), (x_max, y_max)) = pred_boxpts

            # filter out weak detections by ensuring the `confidence`
            # is greater than the minimum confidence
            if pred_conf > args["confidence"]:
                # print prediction to terminal
                print("[INFO] Prediction #{}: confidence={}, "
                      "boxpoints={}".format(i, pred_conf,
                                            pred_boxpts))

                # check if we should show the prediction data
                # on the frame
                if args["display"] > 0:
                    # build a label
                    label = "{}: {:.2f}%".format(CLASSES[label], pred_conf * 100)

                    x_min = int(x_min * W)
                    y_min = int(y_min * H)
                    x_max = int(x_max * W)
                    y_max = int(y_max * H)

                    # extract information from the prediction boxpoints
                    y = y_min - 15 if y_min - 15 > 15 else y_min + 15

                    # display the rectangle and label text
                    cv2.rectangle(image_for_result, (x_min, y_min), (x_max, y_max),
                                  (255, 0, 0), 2)
                    cv2.putText(image_for_result, label, (x_min, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # check to see if we should write the frame to disk
        if writer is not None:
            writer.write(image_for_result)

            # check if we should display the frame on the screen
        # with prediction data (you can achieve faster FPS if you
        # do not output to the screen)
        if args["display"] > 0:
            # display the frame to the screen
            cv2.imshow("Output", image_for_result)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        # update the FPS counter
        fps.update()

    # if "ctrl+c" is pressed in the terminal, break from the loop
    except KeyboardInterrupt:
        break

    # if there's a problem reading a frame, break gracefully
    except AttributeError:
        break

# stop the FPS counter timer
fps.stop()

# destroy all windows if we are displaying them
if args["display"] > 0:
    cv2.destroyAllWindows()

# if we are not using a video file, stop the camera video stream
if not args.get("input", False):
    vs.stop()

# otherwise, release the video file pointer
else:
    vs.release()

# display FPS information
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


