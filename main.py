from __future__ import print_function
from openvino.inference_engine import IENetwork, IEPlugin
from argparse import ArgumentParser, SUPPRESS
from pyimagesearch.customcentroidtracker import CarCentroidTracker, BikeCentroidTracker
from pyimagesearch.customtrackableobject import CarTrackableObject, BikeTrackableObject
from pyimagesearch.lanedetection import lanedetection
from pyimagesearch.lineequation import lineequation
from imutils.video import FPS


import numpy as np
import imutils
import sys
import os
import csv
import cv2
import time
import logging as log

# initialize the class of object to be detected
CLASS_PERSON = 1
CLASS_CAR = 2
CLASS_MOTORBIKE = 3

# initialize our centroid tracker and frame dimensions

(H, W) = (None, None)

lanedetection = lanedetection()
lineequation = lineequation()

carct = CarCentroidTracker()

bikect = BikeCentroidTracker()


cartrackableObjects = {}

biketrackableObjects = {}

totalcount = 0

cartotalminus = 0
cartotalplus = 0

biketotalminus = 0
biketotalplus = 0

status = "[Unavailable]"

detections = 0

writer = None

leftgradient = 0 
leftyintercept = 0
			
rightgradient = 0
rightyintercept = 0

leftx1 = 0
lefty1 = 0
leftx2 = 0
lefty2 = 0

rightx1 = 0
righty1 = 0
rightx2 = 0
righty2 = 0

def build_argparser():
    parser = ArgumentParser(add_help=False)

    args = parser.add_argument_group('Options')
    args.add_argument('-k', '--multiplier', type=int, default=10,
                      help='line multiplier 1/20')
    args.add_argument("-o", "--output", type=str,
	              help="path to optional output video file")
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                      required=True, type=str)
    args.add_argument("-i", "--input",
                      help="Required. Path to video file or image. 'cam' for capturing video stream from camera",
                      required=True, type=str)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is "
                           "acceptable. The demo will look for a suitable plugin for device specified. "
                           "Default value is CPU", default="CPU", type=str)

    return parser


log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
args = build_argparser().parse_args()
model_xml = args.model
model_bin = os.path.splitext(model_xml)[0] + ".bin"

# Plugin initialization for specified device and load extensions library if specified
log.info("Initializing plugin for {} device...".format(args.device))
plugin = IEPlugin(device=args.device, plugin_dirs=None)

# Read IR
log.info("Reading IR...")
net = IENetwork(model=model_xml, weights=model_bin)

if plugin.device == "CPU":
    supported_layers = plugin.get_supported_layers(net)
    not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
    if len(not_supported_layers) != 0:
        log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(plugin.device, ', '.join(not_supported_layers)))
        log.error("Please try to specify cpu extensions library path in demo's command line parameters using -l "
                      "or --cpu_extension command line argument")
        sys.exit(1)

assert len(net.inputs.keys()) == 1, "Demo supports only single input topologies"
assert len(net.outputs) == 1, "Demo supports only single output topologies"

input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))
log.info("Loading IR to the plugin...")
exec_net = plugin.load(network=net, num_requests=2)

# Read and pre-process input image
n, c, h, w = net.inputs[input_blob].shape
del net
if args.input == 'cam':
        input_stream = 1
else:
    input_stream = args.input
    assert os.path.isfile(args.input), "Specified input file doesn't exist"

k = args.multiplier

cap = cv2.VideoCapture(input_stream)

if cap is None or not cap.isOpened():

    cap = cv2.VideoCapture(0)

cur_request_id = 0
next_request_id = 1

log.info("Starting inference in async mode...")
is_async_mode = True

show_stats = True
set_lane_mode = True
flip_mode = False

render_time = 0
ret, frame = cap.read()

print("To close the application, press 'CTRL+C' or any key with focus on the output window")
print("")
print("To flip video output, press 'f' on the output window")
print("")
print("To set or reset lane, press 'Tab' on the output window")
print("")
print("To hide or show stats panel, press 's' on the output window")

fps = FPS().start()

while cap.isOpened():
    
    

    W = int(cap.get(3))
    H = int(cap.get(4))

    center = (W / 2, H / 2)

    if is_async_mode:
        ret, next_frame = cap.read()
        if flip_mode:
            M = cv2.getRotationMatrix2D(center, 180, 1)
            next_frame = cv2.warpAffine(next_frame, M, (W, H))
    else:
        ret, frame = cap.read()
    if not ret:
        break

    resizedframepic = cv2.resize(frame, (512, 288))
    cv2.imwrite("snapshot/1.jpg", resizedframepic) 

    if set_lane_mode:
        canny_image = lanedetection.canny(frame)
        cropped_canny = lanedetection.region_of_interest(canny_image)

        #cv2.imshow("Show Cropped Canny", cropped_canny)
    
        lines = cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 200, np.array([]), minLineLength=300, maxLineGap=10)

        averaged_lines = lanedetection.average_slope_intercept(frame, lines, k)
    	
        line_image = lanedetection.display_lines(frame, averaged_lines)
    else:
        cv2.line(frame, (leftx1, lefty1), (leftx2, lefty2), (0, 255, 255), 10)
        cv2.line(frame, (rightx2, righty2), (rightx1, righty1), (0, 255, 255), 10)
        cv2.line(frame, (leftx2, lefty2), (rightx2, righty2), (0, 255, 255), 10)


    if averaged_lines is not None:
        leftx1 = averaged_lines[0][0][0]
        lefty1 = averaged_lines[0][0][1]
        leftx2 = averaged_lines[0][0][2]
        lefty2 = averaged_lines[0][0][3]
        rightx1 = averaged_lines[1][0][0]
        righty1 = averaged_lines[1][0][1]
        rightx2 = averaged_lines[1][0][2]
        righty2 = averaged_lines[1][0][3]

    rectangle = np.array([[
        (leftx1, lefty1),
        (leftx2, lefty2),
        (rightx2, righty2),
        (rightx1, righty1)]], np.int32)
    cv2.fillPoly(line_image, rectangle, (0, 100, 0), 255)
    
    lineequations = lineequation.getlines(averaged_lines)
	
    if lineequations is not None and lineequations[0] is not None and lineequations[1] is not None:

        leftgradient = lineequations[0][0] 
        leftyintercept = lineequations[0][1]
			
        rightgradient = lineequations[1][0]
        rightyintercept = lineequations[1][1]
			
    initial_w = cap.get(3)
    initial_h = cap.get(4)

        
    if args.output is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args.output, fourcc, 30,(W, H), True)

	
# Main sync point:
# in the truly Async mode we start the NEXT infer request, while waiting for the CURRENT to complete
# in the regular mode we start the CURRENT request and immediately wait for it's completion
    inf_start = time.time()
    if is_async_mode:
        in_frame = cv2.resize(next_frame, (w, h))
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((n, c, h, w))
        exec_net.start_async(request_id=next_request_id, inputs={input_blob: in_frame})
    
    if exec_net.requests[cur_request_id].wait(-1) == 0:
        inf_end = time.time()
        det_time = inf_end - inf_start

        carrects = []
    
        bikerects = []

    # Parse detection results of the current request
        detections = 0
        res = exec_net.requests[cur_request_id].outputs[out_blob]

        for obj in res[0][0]:
            carobjects = []
            bikeobjects = []

            
	     # Draw only objects when probability more than specified threshold
            if obj[2] > 0.5:

                if (int(obj[1]) == (CLASS_CAR or CLASS_MOTORBIKE)):
                    detections += 1
                    
            if obj[2] > 0.7:
                if int(obj[1]) == CLASS_CAR:
                    
                    x1 = int(obj[3] * initial_w)
                    y1 = int(obj[4] * initial_h)
                    x2 = int(obj[5] * initial_w)
                    y2 = int(obj[6] * initial_h)
                     
                    box = np.array([x1, y1, x2, y2])
                    carrects.append(box.astype(int))
                    (startX, startY, endX, endY) = box.astype('int')
                   
                    cv2.rectangle(frame, (startX, startY), (endX,
                           endY), (0, 255, 0), 2) 

                    carobjects = carct.update(carrects)


            if obj[2] > 0.1: 
                if int(obj[1]) == CLASS_MOTORBIKE:

                    x1 = int(obj[3] * initial_w)
                    y1 = int(obj[4] * initial_h)
                    x2 = int(obj[5] * initial_w)
                    y2 = int(obj[6] * initial_h)

                    box = np.array([x1, y1, x2, y2])
                    bikerects.append(box.astype(int))
                    (startX, startY, endX, endY) = box.astype('int')
                    
                    cv2.rectangle(frame, (startX, startY), (endX,
                            endY), (255, 255, 255), 2)
                    
                    bikeobjects = bikect.update(bikerects)

 
        # loop over the tracked objects

            if len(carobjects) != 0:

                for (carobjectID, carcentroid) in \
                    carobjects.items():
      
                    if show_stats:
                        text = 'ID CAR{}'.format(carobjectID)
                        cv2.putText(
                            frame,
                            text,
                            (carcentroid[0] - 10, carcentroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                            )
                    cv2.circle(frame, (carcentroid[0],
                               carcentroid[1]), 4, (0, 255, 0), -1)
                    
                   
                    carto = cartrackableObjects.get(carobjectID,
                            None)

                    if carto is None:
                        carto = CarTrackableObject(carobjectID,
                                carcentroid)
                    else:

                    # the difference between the y-coordinate of the *current*
                    # centroid and the mean of *previous* centroids will tell
                    # us in which direction the object is moving (negative for
                    # 'up' and positive for 'down')

                        y = [c[1] for c in carto.carcentroids]
                        direction = carcentroid[1] - np.mean(y)
                        carto.carcentroids.append(carcentroid)

                        # check to see if the object has been counted or not

                        if not carto.counted:

                        # if the direction is negative (indicating the object
                        # is moving up) AND the centroid is above the center
                        # line, count the object
                        
                            if direction < 0 and carcentroid[1] < lefty2 and carcentroid[0] > leftx2 and carcentroid[0] < rightx2 and np.mean(y) > lefty2:
                        # movingup
                                cartotalplus += 1
                                carto.counted = True
                                cv2.line(frame, (leftx1, lefty1), (leftx2, lefty2), (255, 0, 0), 10)
                                cv2.line(frame, (rightx2, righty2), (rightx1, righty1), (255, 0, 0), 10)
                                cv2.line(frame, (leftx2, lefty2), (rightx2, righty2), (255, 0, 0), 10)

                            elif direction > 0 and carcentroid[1] > lefty2 and carcentroid[0] > leftx2 and carcentroid[0] < rightx2 and np.mean(y) < lefty2:

                        # movingdown

                                cartotalminus += 1
                                carto.counted = True
                                cv2.line(frame, (leftx1, lefty1), (leftx2, lefty2), (255, 0, 0), 10)
                                cv2.line(frame, (rightx2, righty2), (rightx1, righty1), (255, 0, 0), 10)
                                cv2.line(frame, (leftx2, lefty2), (rightx2, righty2), (255, 0, 0), 10)

                  
 
                    cartrackableObjects[carobjectID] = carto

            if len(bikeobjects) != 0:

                for (bikeobjectID, bikecentroid) in \
                    bikeobjects.items():
                
                    if show_stats:
                        text = 'ID BIKE{}'.format(bikeobjectID)
                        cv2.putText(
                            frame,
                            text,
                            (bikecentroid[0] - 10, bikecentroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            2,
                            )
                    cv2.circle(frame, (bikecentroid[0],
                               bikecentroid[1]), 4, (255, 255, 255), -1)
                        


                    biketo = biketrackableObjects.get(bikeobjectID,
                            None)

                    if biketo is None:
                        biketo = BikeTrackableObject(bikeobjectID,
                                bikecentroid)
                    else:

                    # the difference between the y-coordinate of the *current*
                    # centroid and the mean of *previous* centroids will tell
                    # us in which direction the object is moving (negative for
                    # 'up' and positive for 'down')

                        y = [c[1] for c in biketo.bikecentroids]
                        direction = bikecentroid[1] - np.mean(y)
                        biketo.bikecentroids.append(bikecentroid)

                        # check to see if the object has been counted or not

                        if not biketo.counted:

                        # if the direction is negative (indicating the object
                        # is moving up) AND the centroid is above the center
                        # line, count the object
                            
                            if direction < 0 and bikecentroid[1] < lefty2 and bikecentroid[0] > leftx2 and bikecentroid[0] < rightx2 and np.mean(y) > lefty2:

                        #movingup
	
                                biketotalplus += 1
                                biketo.counted = True
                                cv2.line(frame, (leftx1, lefty1), (leftx2, lefty2), (255, 0, 0), 10)
                                cv2.line(frame, (rightx2, righty2), (rightx1, righty1), (255, 0, 0), 10)
                                cv2.line(frame, (leftx2, lefty2), (rightx2, righty2), (255, 0, 0), 10)

                            elif direction > 0 and bikecentroid[1] > lefty2 and bikecentroid[0] > leftx2 and bikecentroid[0] < rightx2 and np.mean(y) < lefty2:

                        # if the direction is positive (indicating the object
                        # is moving down) AND the centroid is below the
                        # center line, count the object

                        #movingdown
                                biketotalminus += 1
                                biketo.counted = True
                                cv2.line(frame, (leftx1, lefty1), (leftx2, lefty2), (255, 0, 0), 10)
                                cv2.line(frame, (rightx2, righty2), (rightx1, righty1), (255, 0, 0), 10)
                                cv2.line(frame, (leftx2, lefty2), (rightx2, righty2), (255, 0, 0), 10)


                    # store the trackable object in our dictionary
 
                    biketrackableObjects[bikeobjectID] = biketo


    totalcount = cartotalplus + cartotalminus + 0 + 0 + biketotalplus + biketotalminus


    if detections > 10:
        status = '[HEAVY]'
    elif detections >= 5 and detections <= 10:
        status = '[MODERATE]'
    elif detections > 0 and detections < 5:
        status = '[LOW]'
    else:
        status = '[NONE]'
    
    if show_stats:
        cv2.putText(frame, "Total Bike Count: " + str(biketotalplus + biketotalminus), (10, H - ((9 * 40) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        cv2.putText(frame, "Total Car Count: " + str(cartotalplus + cartotalminus), (10, H - ((10 * 40) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

        cv2.putText(frame, "TRAFFIC STATUS: " + status, (10, H - ((12 * 40) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
 
        set_lane_message = "[LANE NOT SET] Toggle <Tab> button to set lane" if set_lane_mode else \
                "[LANE IS SET] Toggle <Tab> button to set lane"
        cv2.putText(frame, set_lane_message, (15, 50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255), 3)

    row = [str(cartotalplus) , str(cartotalminus), str(0), str(0), str(biketotalplus), str(biketotalminus), str(detections)]
    with open('Data_File.csv','r') as readfile:
			
        output_reader = csv.reader(readfile)
        lines = list(output_reader)
        lines[0] = row

    with open('Data_File.csv', 'w') as writefile:
        output_writer = csv.writer(writefile)
        output_writer.writerows(lines)
    readfile.close()
    writefile.close()

    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    resizedframe = cv2.resize(combo_image, (640, 360))
    render_start = time.time()

    cv2.imshow("Detection Results", resizedframe)
    
    render_end = time.time()
    render_time = render_end - render_start

    if writer is not None:

        writer.write(combo_image)

    if is_async_mode:
        cur_request_id, next_request_id = next_request_id, cur_request_id
        frame = next_frame

    fps.update()

    key = cv2.waitKey(1)
    if key == 27:
        break

    if (9 == key): #Tab
        set_lane_mode = not set_lane_mode
        log.info("lane is NOT set" if set_lane_mode else "lane is set")

    if (115 == key): #s, show
        show_stats = not show_stats
        log.info("Show Stats Panel" if show_stats else "Hide Stats Panel")

    if (102 == key): #f, flipped
        flip_mode = not flip_mode
        log.info("Video Output is NOT flipped" if flip_mode else "Video Output is flipped")
        if set_lane_mode:
            set_lane_mode = True
            log.info("lane is NOT set" if set_lane_mode else "lane is set")
        else:
            set_lane_mode = not set_lane_mode
            log.info("lane is NOT set" if set_lane_mode else "lane is set")

if writer is not None:
    writer.release()


fps.stop()

row = [str(0) , str(0), str(0), str(0), str(0), str(0), str(0)]
with open('Data_File.csv','r') as readfile:		
    output_reader = csv.reader(readfile)
    lines = list(output_reader)
    lines[0] = row

with open('Data_File.csv', 'w') as writefile:
    output_writer = csv.writer(writefile)
    output_writer.writerows(lines)
readfile.close()
writefile.close()

print("[ INFO ] Elapsed Time: {:.2f}s".format(fps.elapsed()))
print("[ INFO ] Approx. FPS: {:.2f}".format(fps.fps()))

try: 
    os.remove("snapshot/1.jpg")
except: 
    pass

cv2.destroyAllWindows()
