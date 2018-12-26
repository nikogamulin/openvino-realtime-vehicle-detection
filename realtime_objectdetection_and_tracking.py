# USAGE
# python realtime_objectdetection_and_tracking.py --input videos/izola_highway.mp4 --display 1

# import the necessary packages
import os
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import time
import cv2
import numpy as np
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
import dlib

running_on_rpi = False


os_info = os.uname()
if os_info[4][:3] == 'arm':
    running_on_rpi = True


writer = None
W = None
H = None

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=6, maxDistance=100)
trackers = []
trackableObjects = {}

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalDown = 0
totalUp = 0
totalOverall = 0


# initialize the list of class labels our network was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ("background", "aeroplane", "bicycle", "bird",
           "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor")


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


def crop(frame, top, left, height, width):
    h, w, _ = frame.shape
    cropped = frame[top:top + height, left: left + width]
    return cropped



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
ap.add_argument("-s", "--skip-frames", type=int, default=3,
	help="# of skip frames between detections")
args = vars(ap.parse_args())

# Load the model
net = cv2.dnn.readNet('models/mobilenet-ssd/FP16/mobilenet-ssd.xml', 'models/mobilenet-ssd/FP16/mobilenet-ssd.bin')

# Specify target device
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

# if a video path was not supplied, grab a reference to the webcam
if not args.get("input", False):
    print("[INFO] starting video stream...")
    # cap = cv2.VideoCapture(0)
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

        frame = crop(frame, 540, 960, 540, 960)

        # if we are viewing a video and we did not grab a frame then we
        # have reached the end of the video
        if args["input"] is not None and frame is None:
            break

        # the frame from BGR to RGB for dlib
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        H, W, _ = frame.shape
        image_for_result = frame.copy()

        # if we are supposed to be writing a video to disk, initialize
        # the writer
        if args["output"] is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                     (frame.shape[1], frame.shape[0]), True)

        # initialize the current status along with our list of bounding
        # box rectangles returned by either (1) our object detector or
        # (2) the correlation trackers
        status = "Waiting"
        rects = []

        # check to see if we should run a more computationally expensive
        # object detection method to aid our tracker
        if totalFrames % args["skip_frames"] == 0:
            # set the status and initialize our new set of object trackers
            status = "Detecting"
            trackers = []


            # use the NCS to acquire predictions
            predictions = predict(frame, net)

            # loop over our predictions
            for (i, pred) in enumerate(predictions):
                # extract prediction data for readability
                (class_id, pred_conf, pred_boxpts) = pred
                ((x_min, y_min), (x_max, y_max)) = pred_boxpts

                # filter out weak detections by ensuring the `confidence`
                # is greater than the minimum confidence
                if pred_conf > args["confidence"]:
                    # print prediction to terminal
                    print("[INFO] Prediction #{}: confidence={}, "
                          "boxpoints={}".format(i, pred_conf,
                                                pred_boxpts))

                    # extract the index of the class label from the
                    # detections list
                    # idx = int(predictions[0, 0, i, 1])

                    # if the class label is not a person, ignore it
                    if CLASSES[class_id] != "car":
                        continue

                    x_min = int(x_min * W)
                    y_min = int(y_min * H)
                    x_max = int(x_max * W)
                    y_max = int(y_max * H)

                    # construct a dlib rectangle object from the bounding
                    # box coordinates and then start the dlib correlation
                    # tracker
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(x_min, y_min, x_max, y_max)
                    tracker.start_track(rgb, rect)

                    # add the tracker to our list of trackers so we can
                    # utilize it during skip frames
                    trackers.append(tracker)

        # otherwise, we should utilize our object *trackers* rather than
        # object *detectors* to obtain a higher frame processing throughput
        else:
            # loop over the trackers
            for tracker in trackers:
                # set the status of our system to be 'tracking' rather
                # than 'waiting' or 'detecting'
                status = "Tracking"

                # update the tracker and grab the updated position
                tracker.update(rgb)
                pos = tracker.get_position()

                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                # add the bounding box coordinates to the rectangles list
                rects.append((startX, startY, endX, endY))

        # use the centroid tracker to associate the (1) old object
        # centroids with (2) the newly computed object centroids
        objects = ct.update(rects)

        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # check to see if a trackable object exists for the current
            # object ID
            to = trackableObjects.get(objectID, None)

            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objectID, centroid)

            # otherwise, there is a trackable object so we can utilize it
            # to determine direction
            else:
                # the difference between the y-coordinate of the *current*
                # centroid and the mean of *previous* centroids will tell
                # us in which direction the object is moving (negative for
                # 'up' and positive for 'down')
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                # check to see if the object has been counted or not
                if not to.counted:
                    # if the direction is negative (indicating the object
                    # is moving up) AND the centroid is above the center
                    # line, count the object
                    if direction < 0 and centroid[1] < H // 2:
                        totalUp += 1
                        totalOverall += 1
                        to.counted = True

                    # if the direction is positive (indicating the object
                    # is moving down) AND the centroid is below the
                    # center line, count the object
                    elif direction > 0 and centroid[1] > H // 2:
                        totalDown += 1
                        totalOverall += 1
                        to.counted = True


            # store the trackable object in our dictionary
            trackableObjects[objectID] = to

            # build a label
            label = "{}: {:.2f}%".format(CLASSES[class_id], pred_conf * 100)

            # extract information from the prediction boxpoints
            y = y_min - 15 if y_min - 15 > 15 else y_min + 15

            # display the rectangle and label text
            # cv2.rectangle(image_for_result, (x_min, y_min), (x_max, y_max),
            #               (255, 0, 0), 2)
            # cv2.putText(image_for_result, label, (x_min, y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(image_for_result, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(image_for_result, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        # construct a tuple of information we will be displaying on the
        # frame
        info = [
            #("Up", totalUp),
            #("Down", totalDown),
            ("Count", totalOverall),
            ("Status", status),
        ]

        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(image_for_result, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

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

        # increment the total number of frames processed thus far and
        # then update the FPS counter
        totalFrames += 1
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

