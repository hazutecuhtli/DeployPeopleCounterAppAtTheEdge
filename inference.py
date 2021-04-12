#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
"""

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore
import numpy as np

class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.net_pluging = None
        self.infer_request_handle = None        

    def load_model(self, model, device="CPU", cpu_extension=None):
        
        ### TODO: Load the model ###
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"        

        # Initialize the plugin
        self.plugin = IECore()  
        if cpu_extension != None:
            try:
                self.plugin.add_extension(cpu_extension, device)
            except:
                print("This device and extension combinations does not work as expecte.")
                exit(1)
        
        # Read the IR as a IENetwork
        self.network = IENetwork(model=model_xml, weights=model_bin)        
        
        ### TODO: Check for supported layers ###
        supported_layers = self.plugin.query_network(network=self.network, device_name=device)
        
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore.")
            exit(1)        
        
        # Load the IENetwork into the plugin
        self.net_pluging = self.plugin.load_network(network=self.network, device_name=device)        
            
        
        ### TODO: Add any necessary extensions ###
        if (len(unsupported_layers) != 0) and (cpu_extension and "CPU" in device):
            self.core.add_extension(cpu_extension, device)        
        
        ### TODO: Return the loaded inference plugin ###
        ### Note: You may need to update the function parameters. ###
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        
        return self.net_pluging

    def get_input_shape(self):
        
        ### TODO: Return the shape of the input layer ###
        
        return self.network.inputs[self.input_blob].shape

    def exec_net(self, input_image, request_id=0):
        
        ### TODO: Start an asynchronous request ### 
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        self.infer_request_handle = self.net_pluging.start_async(request_id=request_id, inputs={self.input_blob: input_image})
        
        return

    def wait(self, request_id=0):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        status = self.net_pluging.requests[request_id].wait(-1)
        
        return status

    def get_output(self, request_id=0):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        return self.net_pluging.requests[request_id].outputs[self.output_blob]
