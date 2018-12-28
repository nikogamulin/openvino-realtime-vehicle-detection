# USAGE
# python realtime_vehicledetection.py --input videos/izola_highway.mp4 --display 1

running_on_rpi = False
import os

os_info = os.uname()
if os_info[4][:3] == 'arm':
    running_on_rpi = True

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import time
import cv2

writer = None
W = None
H = None

# check if optimization is enabled
if not cv2.useOptimized():
    print("By default, OpenCV has not been optimized")
    cv2.setUseOptimized(True)


def predict(frame, net):
    # Prepare input blob and perform an inference
    blob = cv2.dnn.blobFromImage(frame, size=(672, 384), ddepth=cv2.CV_8U)
    net.setInput(blob)
    out = net.forward()

    predictions = []

    # The net outputs a blob with the shape: [1, 1, N, 7], where N is the number of detected bounding boxes.
    # For each detection, the description has the format: [image_id, label, conf, x_min, y_min, x_max, y_max]

    # Draw detected faces on the frame
    for detection in out.reshape(-1, 7):
        image_id, label, conf, x_min, y_min, x_max, y_max = detection

        if conf > args["confidence"]:
            pred_boxpts = ((x_min, y_min), (x_max, y_max))

            # create prediciton tuple and append the prediction to the
            # predictions list
            prediction = (label, conf, pred_boxpts)
            predictions.append(prediction)

    # return the list of predictions to the calling function
    return predictions

def resize(frame, width, height=None):
    h, w, _ = frame.shape
    if height is None:
        # keep ratio
        factor = width * 1.0/w
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
args = vars(ap.parse_args())

# Load the model
# Description available here: https://github.com/opencv/open_model_zoo/blob/2018/intel_models/vehicle-license-plate-detection-barrier-0106/description/vehicle-license-plate-detection-barrier-0106.md
#net = cv2.dnn.readNet('models/vehicle-detection-adas-0002.xml', 'models/vehicle-detection-adas-0002.bin')

net = cv2.dnn.readNet('models/mobilenet-ssd/mobilenet-ssd.xml', 'models/mobilenet-ssd/mobilenet-ssd.bin')

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
        #frame = resize(frame, 800)
        H, W, _ = frame.shape
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
                    label = "car: {:.2f}%".format(pred_conf * 100)
                    
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
