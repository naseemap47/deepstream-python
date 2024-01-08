#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import sys
sys.path.append('../')
from pathlib import Path
import gi
import configparser
import argparse
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
from ctypes import *
import time
import sys
import math
import platform
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
from common.FPS import PERF_DATA
from common.utils import long_to_uint64
import cv2
import random
import numpy as np
import pyds


no_display = False
silent = False
file_loop = False
perf_data = None

MUXER_OUTPUT_WIDTH=1920
MUXER_OUTPUT_HEIGHT=1080
MUXER_BATCH_TIMEOUT_USEC=4000000
TILED_OUTPUT_WIDTH=1280
TILED_OUTPUT_HEIGHT=720
GST_CAPS_FEATURES_NVMM="memory:NVMM"
OSD_PROCESS_MODE= 0
OSD_DISPLAY_TEXT= 1
MIN_CONFIDENCE = 0.25

class_names = open('labels.txt', 'r').read().splitlines()
colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def draw_bounding_boxes(image, obj_meta, conf, track_id):
    # confidence = '{0:.2f}'.format(confidence)
    rect_params = obj_meta.rect_params
    top = int(rect_params.top)
    left = int(rect_params.left)
    width = int(rect_params.width)
    height = int(rect_params.height)
    obj_name = class_names[obj_meta.class_id]
    plot_one_box(
        [left, top, left+width, top+height], image, 
        colors[obj_meta.class_id], f'{obj_name} {conf:.2} {track_id}', 3
    )
    return image


# tiler_sink_pad_buffer_probe  will extract metadata received on tiler src pad
# and update params for drawing rectangle, object information etc.
def tiler_sink_pad_buffer_probe(pad, info, u_data):
    global feedback
    global global_motor_id
    global success
    global c
    # bolts = 0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # ser.write("Y:1\r".encode())
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        # frame_number = frame_meta.frame_num
        l_obj = frame_meta.obj_meta_list
        # num_rects = frame_meta.num_obj_meta
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            # obj_counter[obj_meta.class_id] += 1
            # if MIN_CONFIDENCE < obj_meta.confidence:
            # print('Batch ID: ', frame_meta.batch_id)
            # OpenCV Image
            n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            # n_frame = draw_bounding_boxes(n_frame, obj_meta, obj_meta.confidence, obj_meta.object_id)
            
            # Tracking
            # print('Track ID: ', obj_meta.object_id)
            track_id = obj_meta.object_id
            # conf = obj_meta.confidence
            rect_params = obj_meta.rect_params
            top = int(rect_params.top)
            left = int(rect_params.left)
            width = int(rect_params.width)
            height = int(rect_params.height)
            class_id = obj_meta.class_id
            
            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        
        # FPS
        stream_index = "stream{0}".format(frame_meta.pad_index)
        global perf_data
        perf_data.update_fps(stream_index)
        
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def cb_newpad(decodebin, decoder_src_pad,data):
    print("In cb_newpad\n")
    caps=decoder_src_pad.get_current_caps()
    if not caps:
        caps = decoder_src_pad.query_caps()
    gststruct=caps.get_structure(0)
    gstname=gststruct.get_name()
    source_bin=data
    features=caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    print("gstname=",gstname)
    if(gstname.find("video")!=-1):
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        print("features=",features)
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad=source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")

def decodebin_child_added(child_proxy,Object,name,user_data):
    print("Decodebin child added:", name, "\n")
    if(name.find("decodebin") != -1):
        Object.connect("child-added",decodebin_child_added,user_data)

    if "source" in name:
        source_element = child_proxy.get_by_name("source")
        if source_element.find_property('drop-on-latency') != None:
            Object.set_property("drop-on-latency", True)



def create_source_bin(index,uri):
    print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name="source-bin-%02d" %index
    print(bin_name)
    nbin=Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    if file_loop:
        # use nvurisrcbin to enable file-loop
        uri_decode_bin=Gst.ElementFactory.make("nvurisrcbin", "uri-decode-bin")
        uri_decode_bin.set_property("file-loop", 1)
        uri_decode_bin.set_property("cudadec-memtype", 0)
    else:
        uri_decode_bin=Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri",uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added",cb_newpad,nbin)
    uri_decode_bin.connect("child-added",decodebin_child_added,nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin,uri_decode_bin)
    bin_pad=nbin.add_pad(Gst.GhostPad.new_no_target("src",Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin

def main(args, requested_pgie=None, config=None, disable_probe=False):
    global perf_data
    perf_data = PERF_DATA(len(args))

    number_sources=len(args)

    # Standard GStreamer initialization
    Gst.init(None)

    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    is_live = False

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    print("Creating streamux \n ")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    pipeline.add(streammux)
    for i in range(number_sources):
        print("Creating source_bin ",i," \n ")
        uri_name=args[i]
        if uri_name.find("rtsp://") == 0 :
            is_live = True
        source_bin=create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname="sink_%u" %i
        sinkpad= streammux.get_request_pad(padname) 
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad=source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)
    nvdslogger = None

    print("Creating Pgie \n ")
    if requested_pgie != None and (requested_pgie == 'nvinferserver' or requested_pgie == 'nvinferserver-grpc') :
        pgie = Gst.ElementFactory.make("nvinferserver", "primary-inference")
    elif requested_pgie != None and requested_pgie == 'nvinfer':
        pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    else:
        pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")

    if not pgie:
        sys.stderr.write(" Unable to create pgie :  %s\n" % requested_pgie)

    if disable_probe:
        # Use nvdslogger for perf measurement instead of probe function
        print ("Creating nvdslogger \n")
        nvdslogger = Gst.ElementFactory.make("nvdslogger", "nvdslogger")

    print("Creating nvtracker \n ")
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    if not tracker:
        sys.stderr.write(" Unable to create tracker \n")

    # Add nvvidconv1 and filter1 to convert the frames to RGBA
    # which is easier to work with in Python.

    print("Creating nvvidconv1 \n ")
    nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
    if not nvvidconv1:
        sys.stderr.write(" Unable to create nvvidconv1 \n")
    print("Creating filter1 \n ")
    caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    filter1 = Gst.ElementFactory.make("capsfilter", "filter1")
    if not filter1:
        sys.stderr.write(" Unable to get the caps filter1 \n")
    filter1.set_property("caps", caps1)

    print("Creating tiler \n ")
    tiler=Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write(" Unable to create tiler \n")
    print("Creating nvvidconv \n ")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")
    print("Creating nvosd \n ")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")
    nvosd.set_property('process-mode',OSD_PROCESS_MODE)
    nvosd.set_property('display-text',OSD_DISPLAY_TEXT)

    if file_loop:
        if is_aarch64():
            # Set nvbuf-memory-type=4 for aarch64 for file-loop (nvurisrcbin case)
            streammux.set_property('nvbuf-memory-type', 0)
        else:
            # Set nvbuf-memory-type=2 for x86 for file-loop (nvurisrcbin case)
            streammux.set_property('nvbuf-memory-type', 2)

    if no_display:
        print("Creating Fakesink \n")
        sink = Gst.ElementFactory.make("fakesink", "fakesink")
        sink.set_property('enable-last-sample', 0)
        sink.set_property('sync', 0)
    else:
        if is_aarch64():
            print("Creating nv3dsink \n")
            sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
            if not sink:
                sys.stderr.write(" Unable to create nv3dsink \n")
        else:
            print("Creating EGLSink \n")
            sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
            if not sink:
                sys.stderr.write(" Unable to create egl sink \n")

    if not sink:
        sys.stderr.write(" Unable to create sink element \n")

    if is_live:
        print("At least one of the sources is live")
        streammux.set_property('live-source', 1)

    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', number_sources)
    streammux.set_property('batched-push-timeout', 4000000)
    if requested_pgie == "nvinferserver" and config != None:
        pgie.set_property('config-file-path', config)
    elif requested_pgie == "nvinferserver-grpc" and config != None:
        pgie.set_property('config-file-path', config)
    elif requested_pgie == "nvinfer" and config != None:
        pgie.set_property('config-file-path', config)
    else:
        pgie.set_property('config-file-path', "config_infer_primary_yoloV8.txt")
    pgie_batch_size=pgie.get_property("batch-size")
    if(pgie_batch_size != number_sources):
        print("WARNING: Overriding infer-config batch-size",pgie_batch_size," with number of sources ", number_sources," \n")
        pgie.set_property("batch-size",number_sources)
    tiler_rows=int(math.sqrt(number_sources))
    tiler_columns=int(math.ceil((1.0*number_sources)/tiler_rows))
    tiler.set_property("rows",tiler_rows)
    tiler.set_property("columns",tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)
    sink.set_property("qos",0)

    config = configparser.ConfigParser()
    config.read('tracker_config.txt')
    config.sections()

    for key in config['tracker']:
        if key == 'tracker-width':
            tracker_width = config.getint('tracker', key)
            tracker.set_property('tracker-width', tracker_width)
        if key == 'tracker-height':
            tracker_height = config.getint('tracker', key)
            tracker.set_property('tracker-height', tracker_height)
        if key == 'gpu-id':
            tracker_gpu_id = config.getint('tracker', key)
            tracker.set_property('gpu_id', tracker_gpu_id)
        if key == 'll-lib-file':
            tracker_ll_lib_file = config.get('tracker', key)
            tracker.set_property('ll-lib-file', tracker_ll_lib_file)
        if key == 'll-config-file':
            tracker_ll_config_file = config.get('tracker', key)
            tracker.set_property('ll-config-file', tracker_ll_config_file)
        if key == 'enable-batch-process':
            tracker_enable_batch_process = config.getint('tracker', key)
            tracker.set_property('enable_batch_process', tracker_enable_batch_process)
        if key == 'enable-past-frame':
            tracker_enable_past_frame = config.getint('tracker', key)
            tracker.set_property('enable_past_frame', tracker_enable_past_frame)

    print("Adding elements to Pipeline \n")
    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(tiler)
    pipeline.add(nvvidconv)
    pipeline.add(filter1)
    pipeline.add(nvvidconv1)
    pipeline.add(nvosd)
    pipeline.add(sink)

    print("Linking elements in the Pipeline \n")
    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(nvvidconv1)
    nvvidconv1.link(filter1)
    filter1.link(tiler)
    tiler.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(sink) 

    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop) 

    tiler_sink_pad = tiler.get_static_pad("sink")
    if not tiler_sink_pad:
        sys.stderr.write(" Unable to get src pad \n")
    else:
        tiler_sink_pad.add_probe(Gst.PadProbeType.BUFFER, tiler_sink_pad_buffer_probe, 0)
        # perf callback function to print fps every 5 sec
        GLib.timeout_add(5000, perf_data.perf_print_callback)

    # List the sources
    print("Now playing...")
    for i, source in enumerate(args):
        print(i, ": ", source)

    print("Starting pipeline \n")
    # start play back and listed to events		
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    print("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)

def parse_args():

    parser = argparse.ArgumentParser(prog="deepstream_test_3",
                    description="deepstream-test3 multi stream, multi model inference reference app")
    parser.add_argument(
        "-i",
        "--input",
        help="Path to input streams",
        nargs="+",
        metavar="URIs",
        default=[
            "rtsp://admin:eternaler4444@192.168.29.185:554/cam/realmonitor?channel=1&subtype=0",
            "rtsp://admin:eternaler4444@192.168.29.185:554/cam/realmonitor?channel=1&subtype=0",
            "rtsp://admin:eternaler4444@192.168.29.185:554/cam/realmonitor?channel=1&subtype=0",
            "rtsp://admin:eternaler4444@192.168.29.185:554/cam/realmonitor?channel=1&subtype=0",
            # "file:///home/orin/Videos/1.mp4",
            # "file:///home/orin/Videos/2.mp4",
            # "file:///home/orin/Videos/3.mp4",
            # "file:///home/orin/Videos/4.mp4",
        ],
        required=False,
    )
    parser.add_argument(
        "-c",
        "--configfile",
        metavar="config_location.txt",
        default=None,
        help="Choose the config-file to be used with specified pgie",
    )
    parser.add_argument(
        "-g",
        "--pgie",
        default=None,
        help="Choose Primary GPU Inference Engine",
        choices=["nvinfer", "nvinferserver", "nvinferserver-grpc"],
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        default=False,
        dest='no_display',
        help="Disable display of video output",
    )
    parser.add_argument(
        "--file-loop",
        action="store_true",
        default=False,
        dest='file_loop',
        help="Loop the input file sources after EOS",
    )
    parser.add_argument(
        "--disable-probe",
        action="store_true",
        default=False,
        dest='disable_probe',
        help="Disable the probe function and use nvdslogger for FPS",
    )
    parser.add_argument(
        "-s",
        "--silent",
        action="store_true",
        default=False,
        dest='silent',
        help="Disable verbose output",
    )
    args = parser.parse_args()

    stream_paths = args.input
    pgie = args.pgie
    config = args.configfile
    disable_probe = args.disable_probe
    global no_display
    global silent
    global file_loop
    no_display = args.no_display
    silent = args.silent
    file_loop = args.file_loop

    if config and not pgie or pgie and not config:
        sys.stderr.write ("\nEither pgie or configfile is missing. Please specify both! Exiting...\n\n\n\n")
        parser.print_help()
        sys.exit(1)
    if config:
        config_path = Path(config)
        if not config_path.is_file():
            sys.stderr.write ("Specified config-file: %s doesn't exist. Exiting...\n\n" % config)
            sys.exit(1)

    print(vars(args))
    return stream_paths, pgie, config, disable_probe

if __name__ == '__main__':
    stream_paths, pgie, config, disable_probe = parse_args()
    sys.exit(main(stream_paths, pgie, config, disable_probe))
