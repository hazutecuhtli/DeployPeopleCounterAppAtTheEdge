"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.554 -c GREEN | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
"""

import os
import sys
import time
import socket
import json
import cv2
import numpy as np
import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60
#CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')    
    
    required.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    required.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    # Note - CPU extensions are moved to plugin since OpenVINO release 2020.1. 
    # The extensions are loaded automatically while     
    # loading the CPU plugin, hence 'add_extension' need not be used.

    #required.add_argument("-l", "--cpu_extension", required=False, type=str,
    #                         default=None,
    #                         help="MKLDNN (CPU)-targeted custom layers."
    #                              "Absolute path to a shared library with the"
    #                              "kernels impl.")
    
    optional.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    
    optional.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    optional.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    optional.add_argument("-c", "--Color", help='c_desc', default='BLUE')    
    
    args = parser.parse_args()
    
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    
    return client

def capture_stream(args, valid_exts, size, CODEC = 0x00000021):

    ### Handle image, video or webcam
    # Create a flag for single images
    image_flag = False
    # Check if the input is a webcam
    if (args.input == 'CAM') or (args.input[-4:] in valid_exts['video']):
        args.input = 0
    elif args.input[-4:] in valid_exts['imgs']:
        image_flag = True


        
    ### Get and open video capture
    path = os.path.join(os.getcwd(), "outputs")
    if image_flag:
        output = None
    else:
        output = cv2.VideoWriter(os.path.join(path , 'output_video.mp4'), 
                                     CODEC, 30, (size[0], size[1]))        
        
    return output, image_flag
    
def draw_boxes(frame, result, args, width, height, time):
    '''
    Draw bounding boxes onto the frame.
    '''
    #Persons counter
    counter = 0
    
    for box in result[0][0]: # Output shape is 1x1xNx7
        conf = box[2]
        
        if conf >= args.prob_threshold:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), convert_color(args.Color), 1)
            counter += 1
    
    return frame, counter
    
    
def convert_color(color_string):
    '''
    Get the BGR value of the desired bounding box color.
    Defaults to Blue if an invalid color is given.
    '''
    colors = {"BLUE": (255,0,0), "GREEN": (0,255,0), "RED": (0,0,255)}
    out_color = colors.get(color_string)
    if out_color:
        return out_color
    else:
        return colors['BLUE']    
          
def infer_on_stream(args, client, frames_variance=10):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    
    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, args.cpu_extension)

    ### TODO: Handle the input stream ###
    # Creating the output
    valid_exts = {'video':['.mp4'], 'imgs':['.jpg', '.bmp']} 

    if (args.input[-4:] not in valid_exts['video']) and (args.input[-4:] not in valid_exts['imgs']) and (args.input != 'CAM'):
        print("The selected input is no valid to be used on this program, try a different one.")
        exit(1)
    
    # Get and open video capture
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))    

    output, single_image = capture_stream(args, valid_exts, size = (width, height))
    
    #start_time
    start_time = time.time()
    frame_static = 0
    total_count = 0
    duration = 0
    times_presence = []
    time_person = 0
    flag_person = False
    waiting = '.'
    
    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()

        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        
        ### TODO: Pre-process the image as needed ###
        net_input_shape = infer_network.get_input_shape()
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)        

        ### TODO: Start asynchronous inference for specified request ###
        com_start_time = time.time()
        infer_network.exec_net(p_frame)

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
        
            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output()
            tiempo_inicio = time.time()
            #print('aquiiiii', "{:.09f}".format(time.time()-tiempo_inicio), result[0][0])
            ### TODO: Extract any desired stats from the results ###
            frame, current_count = draw_boxes(frame, result, args, width, height, com_start_time)
            count_temp = current_count

            ### TODO: Calculate and send relevant information on ###
            
            # Code to smooth the count of persons considering the frame_variance
            # numbers of frames
            if (frame_static == 0) or (frame_static == frames_variance):
                frame_static = 0
                if total_count > 0 and current_count > np.sum(persons_in_frame):
                    current_count -= np.sum(persons_in_frame)
                persons_in_frame = [current_count]
            elif frame_static < frames_variance:
                if current_count > sum(persons_in_frame):
                    current_count -= sum(persons_in_frame)
                persons_in_frame.append(current_count)
                
            # Code to calculate the total number of persons found in the analyzed
            # input used 
            if (0 not in persons_in_frame) and not flag_person:  
                time_person = time.time()
                total_count += 1
                # Publishing the total count variable
                client.publish("person", json.dumps({"total": total_count}))      
                flag_person = True
            elif ((current_count + sum(persons_in_frame)) == 0) and flag_person:
                times_presence.append(time.time()-time_person)
                duration = np.mean(times_presence)
                # Publishing the average person time variable
                client.publish("person/duration", json.dumps({"duration": duration})) 
                flag_person = False

            # Selecting the next frame to analyze
            frame_static += 1           

            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            # Publishing the count variable
            client.publish("person", json.dumps({"count": current_count}))
                    
            ### TODO: Send the frame to the FFMPEG server ###
            sys.stdout.buffer.write(frame)
            sys.stdout.flush()

            ### TODO: Write an output image if `single_image_mode` ###
            if single_image:
                path = os.path.join(os.getcwd(), "outputs")
                cv2.imwrite(os.path.join(path ,'output_image.jpg'), frame)
            else:
                output.write(frame) 
            
        # Break if escape key pressed
        if key_pressed == 27:
            break        
     
    # Release all captured frames and destroy any openCVwindows
    cap.release()
    cv2.destroyAllWindows()
    
    #Disconnect from MQTT
    client.disconnect()
            
def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)

if __name__ == '__main__':
    main()
