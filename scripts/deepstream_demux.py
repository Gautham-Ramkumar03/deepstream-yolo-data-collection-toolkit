import sys

sys.path.append("../")
import gi
import configparser
import argparse

gi.require_version("Gst", "1.0")
from gi.repository import Gst
from gi.repository import GLib
from ctypes import *
import time
import sys
import os
import math
import platform
from common.platform_info import PlatformInfo
from common.bus_call import bus_call
from common.FPS import PERF_DATA

import pyds
import csv
import cv2  # Add the missing OpenCV import
from datetime import datetime
import os
import numpy as np
from threading import Thread, Lock, Event

# Import the structured saver
from structured_saver_1 import InferenceSaver
# Import the frame grabber
from frame_grabber import FrameGrabber

no_display = False
silent = False
file_loop = False
perf_data = None
inference_saver = None  # Global reference to InferenceSaver
frame_grabber = None  # Global reference to FrameGrabber
global_input_sources = None
draw_bounding_boxes = False  # Global flag for bounding box drawing

MAX_DISPLAY_LEN = 64
MUXER_OUTPUT_WIDTH = 540
MUXER_OUTPUT_HEIGHT = 540  # 1080
MUXER_BATCH_TIMEOUT_USEC = 33000
TILED_OUTPUT_WIDTH = 640  # 1280
TILED_OUTPUT_HEIGHT = 360  # 720
GST_CAPS_FEATURES_NVMM = "memory:NVMM"
OSD_PROCESS_MODE = 0
OSD_DISPLAY_TEXT = 1

def pgie_src_pad_buffer_probe(pad, info, u_data):
    """
    The function pgie_src_pad_buffer_probe() is a callback function that is called every time a buffer
    is received on the source pad of the pgie element. 
    Now also saves detection results to the structured directory using InferenceSaver.
    Handles all 80 COCO classes.
    """
    frame_number = 0
    num_rects = 0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return
    
    # Get global references
    global inference_saver, frame_grabber, draw_bounding_boxes
    
    # Retrieve batch metadata from the gst_buffer
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    
    # Process each frame in the batch
    while l_frame is not None:
        try:
            # Cast frame metadata
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        stream_id = frame_meta.pad_index
        l_obj = frame_meta.obj_meta_list
        num_rects = frame_meta.num_obj_meta
        
        # Count objects by class for terminal output - initialize for all 80 COCO classes
        obj_counter = {class_id: 0 for class_id in range(80)}
        
        # Get the clean frame from our frame grabber (already clean, no bboxes)
        print(f"\n--- Processing frame {frame_number} from stream {stream_id} ---")
        clean_frame = frame_grabber.get_frame(stream_id) if frame_grabber else None
        
        # Check if we got a valid frame
        if clean_frame is None:
            print(f"Warning: No frame available for stream {stream_id}, frame {frame_number}")
            print("Using black frame as fallback...")
            clean_frame = frame_grabber.get_black_frame(stream_id)
            
            if num_rects > 0:
                print(f"NOTICE: Using black frame for {num_rects} objects!")
        
        # Collect all detections for this frame first
        frame_detections = []
        l_obj_temp = l_obj
        
        while l_obj_temp is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj_temp.data)
            except StopIteration:
                break
                
            # Update object counter for terminal output (handle all classes 0-79)
            if 0 <= obj_meta.class_id < 80:
                obj_counter[obj_meta.class_id] += 1
            
            # Get bounding box coordinates
            rect_params = obj_meta.rect_params
            
            # Get precise timestamp for this detection
            detection_time = datetime.now()
            detection_timestamp = detection_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            detection_date = detection_time.strftime("%Y-%m-%d")
            detection_time_str = detection_time.strftime("%H:%M:%S.%f")[:-3]
            
            # Get class name from inference_saver mapping
            class_name = inference_saver.CLASS_NAMES.get(obj_meta.class_id, f"Unknown_{obj_meta.class_id}")
            
            # Store detection info
            detection_info = {
                'detection_data': {
                    'timestamp': detection_timestamp,
                    'date': detection_date,
                    'time': detection_time_str,
                    'stream_id': stream_id,
                    'frame_num': frame_number,
                    'obj_id': obj_meta.object_id,
                    'class_id': obj_meta.class_id,
                    'class_name': class_name,
                    'left': rect_params.left,
                    'top': rect_params.top,
                    'width': rect_params.width,
                    'height': rect_params.height,
                    'confidence': obj_meta.confidence
                },
                'bbox': {
                    'left': int(rect_params.left),
                    'top': int(rect_params.top),
                    'width': int(rect_params.width),
                    'height': int(rect_params.height),
                    'class_id': obj_meta.class_id,
                    'class_name': class_name
                }
            }
            
            frame_detections.append(detection_info)
            
            try:
                l_obj_temp = l_obj_temp.next
            except StopIteration:
                break
        
        # Now save each detection individually using the same clean frame
        for detection_info in frame_detections:
            detection_data = detection_info['detection_data']
            
            # Save the detection using InferenceSaver - pass clean_frame for both parameters
            # The structured_saver will handle bounding box drawing if needed
            inference_saver.save_detection(
                detection_data, 
                stream_id, 
                detection_data['class_id'], 
                frame_number, 
                detection_data['obj_id'],
                clean_frame,  # Use clean frame as "original"
                clean_frame,  # Use clean frame as "inferenced" - structured_saver handles bbox drawing
                source_uri=global_input_sources[stream_id] if global_input_sources else None
            )
        
        # Also save all detections together in the detections folder
        if frame_detections:
            camera_name = inference_saver.map_source_to_camera_name(stream_id, 
                global_input_sources[stream_id] if global_input_sources else None)
            
            # Save combined frame detections using clean frame
            inference_saver.save_frame_detections(
                frame_detections,
                camera_name,
                clean_frame,  # Use clean frame - structured_saver handles bbox drawing
                save_type="inferenced" if draw_bounding_boxes else "original"
            )

        # Terminal output for object counts - show only classes that were detected
        detected_classes = [(class_id, count) for class_id, count in obj_counter.items() if count > 0]
        
        if detected_classes:
            # Show frame info and total object count
            print(f"Frame Number={frame_number}, Number of Objects={num_rects}")
            
            # Show detected classes with their names and counts
            detection_summary = []
            for class_id, count in detected_classes:
                class_name = inference_saver.CLASS_NAMES.get(class_id, f"class_{class_id}")
                detection_summary.append(f"{class_name}={count}")
            
            # Print detected classes in a readable format
            print(f"Detected classes: {', '.join(detection_summary)}")
        else:
            print(f"Frame Number={frame_number}, Number of Objects={num_rects} (no valid detections)")

        # Update frame rate through this probe
        stream_index = "stream{0}".format(frame_meta.pad_index)
        global perf_data
        perf_data.update_fps(stream_index)
        
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def cb_newpad(decodebin, decoder_src_pad, data):
    """
    The function is called when a new pad is created by the decodebin. 
    The function checks if the new pad is for video and not audio. 
    If the new pad is for video, the function checks if the pad caps contain NVMM memory features. 
    If the pad caps contain NVMM memory features, the function links the decodebin pad to the source bin
    ghost pad. 
    If the pad caps do not contain NVMM memory features, the function prints an error message.
    :param decodebin: The decodebin element that is creating the new pad
    :param decoder_src_pad: The source pad created by the decodebin element
    :param data: This is the data that was passed to the callback function. In this case, it is the
    source_bin
    """
    print("In cb_newpad\n")
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    print("gstname=", gstname)
    if gstname.find("video") != -1:
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        print("features=", features)
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write(
                    "Failed to link decoder src pad to source bin ghost pad\n"
                )
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")


def decodebin_child_added(child_proxy, Object, name, user_data):
    """
    If the child added to the decodebin is another decodebin, connect to its child-added signal. If the
    child added is a source, set its drop-on-latency property to True.
    
    :param child_proxy: The child element that was added to the decodebin
    :param Object: The object that emitted the signal
    :param name: The name of the element that was added
    :param user_data: This is a pointer to the data that you want to pass to the callback function
    """
    print("Decodebin child added:", name, "\n")
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)

    if "source" in name:
        source_element = child_proxy.get_by_name("source")
        if source_element.find_property("drop-on-latency") != None:
            Object.set_property("drop-on-latency", True)


def create_source_bin(index, source_uri):
    """
    Creates a source bin for handling different types of input sources.
    Supports USB/CSI cameras.
    
    :param index: The index of the source for unique naming
    :param source_uri: URI of the source (/dev/video{n}, etc.)
    :return: A configured source bin for the pipeline
    """
    bin_name = f"source-bin-{index:02d}"
    print(f"Creating {bin_name} for source: {source_uri}")
    
    # Register camera with inference_saver using simplified naming
    if inference_saver:
        camera_name = inference_saver.register_camera(index, source_uri)
        print(f"Registered camera {index} as '{camera_name}'")
    
    # Create bin
    bin = Gst.Bin.new(bin_name)
    if not bin:
        sys.stderr.write(f" Unable to create source bin {bin_name}\n")
        return None

    if source_uri.startswith("/dev/video"):
        # USB/CSI mono camera via v4l2src
        src = Gst.ElementFactory.make("v4l2src", f"v4l2src-{index}")
        if not src:
            sys.stderr.write(f"Error: Unable to create v4l2src element\n")
            return None
            
        src.set_property("device", source_uri)
        
        # Create videoconvert for format conversion
        conv = Gst.ElementFactory.make("videoconvert", f"conv-{index}")
        if not conv:
            sys.stderr.write(f"Error: Unable to create videoconvert element\n")
            return None
            
        # Create nvvideoconvert for NVMM memory
        nvconv = Gst.ElementFactory.make("nvvideoconvert", f"nvconv-{index}")
        if not nvconv:
            sys.stderr.write(f"Error: Unable to create nvvideoconvert element\n")
            return None
            
        # Create capsfilter for format constraints
        caps = Gst.ElementFactory.make("capsfilter", f"caps-{index}")
        if not caps:
            sys.stderr.write(f"Error: Unable to create capsfilter element\n")
            return None
            
        caps.set_property("caps",
            Gst.Caps.from_string("video/x-raw(memory:NVMM),format=NV12,framerate=30/1"))
        
        # Add all elements to bin
        bin.add(src)
        bin.add(conv)
        bin.add(nvconv)
        bin.add(caps)
        
        # Link elements within bin
        if not src.link(conv):
            sys.stderr.write(f"Error: Failed to link v4l2src to videoconvert\n")
            return None
            
        if not conv.link(nvconv):
            sys.stderr.write(f"Error: Failed to link videoconvert to nvvideoconvert\n")
            return None
            
        if not nvconv.link(caps):
            sys.stderr.write(f"Error: Failed to link nvvideoconvert to capsfilter\n")
            return None
        
        # Create ghost pad to expose bin's output
        ghost = Gst.GhostPad.new("src", caps.get_static_pad("src"))
        if not ghost:
            sys.stderr.write(f"Error: Failed to create ghost pad\n")
            return None
            
        bin.add_pad(ghost)

    else:
        sys.stderr.write(f"Error: Unknown source URI format: {source_uri}\n")
        return None

    print(f"Successfully created {bin_name}")
    return bin

def make_element(element_name, i):
    """
    Creates a Gstreamer element with unique name
    Unique name is created by adding element type and index e.g. `element_name-i`
    Unique name is essential for all the element in pipeline otherwise gstreamer will throw exception.
    :param element_name: The name of the element to create
    :param i: the index of the element in the pipeline
    :return: A Gst.Element object
    """
    element = Gst.ElementFactory.make(element_name, element_name)
    if not element:
        sys.stderr.write(" Unable to create {0}".format(element_name))
    element.set_property("name", "{0}-{1}".format(element_name, str(i)))
    return element


def main(args, requested_pgie=None, config=None, disable_probe=False):
    input_sources = args
    number_sources = len(input_sources)
    global perf_data, inference_saver, frame_grabber, draw_bounding_boxes
    perf_data = PERF_DATA(number_sources)
    global global_input_sources
    global_input_sources = input_sources
    
    # Initialize the structured saver with bbox flag and dataset mode
    output_dir = "data"  # Base directory for all outputs
    inference_saver = InferenceSaver(
        base_dir=output_dir,
        enable_bbox_similarity_filtering=True,
        iou_threshold=0.5,
        memory_seconds=30,
        draw_bboxes=draw_bounding_boxes,  # Pass the bbox flag
        dataset_mode=dataset_mode  # Pass the dataset flag
    )
    print(f"Initialized structured output saving to {output_dir}")
    if dataset_mode:
        print("DATASET MODE: Creating YOLO format dataset structure")
    else:
        print(f"Will create cameras: {', '.join([f'cam{i}' for i in range(number_sources)])}")
    
    # Initialize the frame grabber
    frame_grabber = FrameGrabber()
    print("Initialized frame grabber")
    
    # Print bounding box status
    if draw_bounding_boxes:
        print("Bounding box drawing: ENABLED")
    else:
        print("Bounding box drawing: DISABLED")

    platform_info = PlatformInfo()
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

    # If not on integrated GPU (Jetson), use CUDA unified memory for easier CPU access
    if not platform_info.is_integrated_gpu():
        # Use CUDA unified memory in the pipeline so frames
        # can be easily accessed on CPU in Python.
        mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        streammux.set_property("nvbuf-memory-type", mem_type)

    pipeline.add(streammux)
    for i in range(number_sources):
        print("Creating source_bin ", i, " \n ")
        uri_name = input_sources[i]
        if uri_name.find("rtsp://") == 0:
            is_live = True
        source_bin = create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname = "sink_%u" % i
        sinkpad = streammux.request_pad_simple(padname)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)

    queue1 = Gst.ElementFactory.make("queue", "queue1")
    pipeline.add(queue1)
    print("Creating Pgie \n ")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")

    print("Creating nvstreamdemux \n ")
    nvstreamdemux = Gst.ElementFactory.make("nvstreamdemux", "nvstreamdemux")
    if not nvstreamdemux:
        sys.stderr.write(" Unable to create nvstreamdemux \n")

    if is_live:
        print("Atleast one of the sources is live")
        streammux.set_property("live-source", 1)

    streammux.set_property("width", 640)
    streammux.set_property("height", 480)
    streammux.set_property("batch-size", number_sources)
    streammux.set_property("batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC)
    pgie.set_property("config-file-path", "ds_demux_pgie_config.txt")
    pgie_batch_size = pgie.get_property("batch-size")
    if pgie_batch_size != number_sources:
        print(
            "WARNING: Overriding infer-config batch-size",
            pgie_batch_size,
            " with number of sources ",
            number_sources,
            " \n",
        )
        pgie.set_property("batch-size", number_sources)

    print("Adding elements to Pipeline \n")
    pipeline.add(pgie)
    pipeline.add(nvstreamdemux)

    # linking
    streammux.link(queue1)
    queue1.link(pgie)
    pgie.link(nvstreamdemux)
    ##creating demux src

    for i in range(number_sources):
        # pipeline nvstreamdemux -> queue -> nvvidconv -> tee -> [nvosd -> sink] & [clean capture branch]
        # Creating EGLsink
        if platform_info.is_integrated_gpu():
            print("Creating nv3dsink \n")
            sink = make_element("fakesink", i)
            if not sink:
                sys.stderr.write(" Unable to create nv3dsink \n")
        else:
            if platform_info.is_platform_aarch64():
                print("Creating nv3dsink \n")
                sink = make_element("fakesink", i)
            else:
                print("Creating EGLSink \n")
                sink = make_element("nveglglessink", i)
            if not sink:
                sys.stderr.write(" Unable to create egl sink \n")
        pipeline.add(sink)

        # creating queue
        queue = make_element("queue", i)
        pipeline.add(queue)

        # creating nvvidconv
        nvvideoconvert = make_element("nvvideoconvert", i)
        pipeline.add(nvvideoconvert)

        # Create tee to split the stream after nvvideoconvert (before nvosd)
        tee = Gst.ElementFactory.make("tee", f"tee-{i}")
        if not tee:
            sys.stderr.write(f" Unable to create tee-{i}\n")
            continue

        # creating nvosd
        nvdsosd = make_element("nvdsosd", i)
        pipeline.add(nvdsosd)
        nvdsosd.set_property("process-mode", OSD_PROCESS_MODE)
        nvdsosd.set_property("display-text", OSD_DISPLAY_TEXT)

        # Create queue for the display branch
        queue_display = Gst.ElementFactory.make("queue", f"queue-display-{i}")
        if not queue_display:
            sys.stderr.write(f" Unable to create queue-display-{i}\n")
            continue

        # Create queue for the clean frame branch
        queue_clean = Gst.ElementFactory.make("queue", f"queue-clean-{i}")
        if not queue_clean:
            sys.stderr.write(f" Unable to create queue-clean-{i}\n")
            continue

        # Add nvvideoconvert for clean frame capture (RGBA conversion)
        print(f"Creating nvvideoconvert-clean for clean frame capture on stream {i}")
        nvvideoconvert_clean = Gst.ElementFactory.make("nvvideoconvert", f"nvvideoconvert-clean-{i}")
        if not nvvideoconvert_clean:
            sys.stderr.write(f" Unable to create nvvideoconvert-clean-{i}\n")
            continue

        # Set memory type based on platform for clean frame converter
        if not platform_info.is_integrated_gpu():
            mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
            nvvideoconvert_clean.set_property("nvbuf-memory-type", mem_type)
            
        # Create capsfilter for clean RGBA format
        capsfilter_clean = Gst.ElementFactory.make("capsfilter", f"capsfilter-clean-{i}")
        if not capsfilter_clean:
            sys.stderr.write(f" Unable to create capsfilter-clean-{i}\n")
            continue
            
        # Set RGBA caps with NVMM memory for clean frames
        caps_clean = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
        capsfilter_clean.set_property("caps", caps_clean)

        # Create fakesink for the clean frame branch (we only want to capture frames, not display)
        fakesink_clean = Gst.ElementFactory.make("fakesink", f"fakesink-clean-{i}")
        if not fakesink_clean:
            sys.stderr.write(f" Unable to create fakesink-clean-{i}\n")
            continue
        
        # Add all elements to pipeline
        pipeline.add(tee)
        pipeline.add(queue_display)
        pipeline.add(queue_clean)
        pipeline.add(nvvideoconvert_clean)
        pipeline.add(capsfilter_clean)
        pipeline.add(fakesink_clean)

        # connect nvstreamdemux -> queue
        padname = "src_%u" % i
        demuxsrcpad = nvstreamdemux.request_pad_simple(padname)
        if not demuxsrcpad:
            sys.stderr.write("Unable to create demux src pad \n")

        queuesinkpad = queue.get_static_pad("sink")
        if not queuesinkpad:
            sys.stderr.write("Unable to create queue sink pad \n")
        demuxsrcpad.link(queuesinkpad)

        # Link the main path: queue -> nvvideoconvert -> tee
        queue.link(nvvideoconvert)
        nvvideoconvert.link(tee)
        
        # Branch 1: Display path with nvosd
        # tee -> queue_display -> nvosd -> sink
        tee_display_pad = tee.request_pad_simple("src_%u")
        queue_display_sink_pad = queue_display.get_static_pad("sink")
        if not tee_display_pad or not queue_display_sink_pad:
            sys.stderr.write(f"Unable to get display branch pads for stream {i}\n")
            continue
        tee_display_pad.link(queue_display_sink_pad)
        queue_display.link(nvdsosd)
        nvdsosd.link(sink)
        
        # Branch 2: Clean frame capture path
        # tee -> queue_clean -> nvvideoconvert_clean -> capsfilter_clean -> fakesink_clean
        tee_clean_pad = tee.request_pad_simple("src_%u")
        queue_clean_sink_pad = queue_clean.get_static_pad("sink")
        if not tee_clean_pad or not queue_clean_sink_pad:
            sys.stderr.write(f"Unable to get clean branch pads for stream {i}\n")
            continue
        tee_clean_pad.link(queue_clean_sink_pad)
        queue_clean.link(nvvideoconvert_clean)
        nvvideoconvert_clean.link(capsfilter_clean)
        capsfilter_clean.link(fakesink_clean)

        sink.set_property("qos", 0)
        sink.set_property("sync", 0)  # Disable sync for maximum performance
        
        fakesink_clean.set_property("qos", 0)
        fakesink_clean.set_property("sync", 0)  # Disable sync for maximum performance

    print("Linking elements in the Pipeline \n")
    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)
    
    # Instead of attaching probe to pgie source pad, attach to nvosd sink pad for each stream
    # This follows the NVIDIA example pattern and ensures better synchronization
    print("Setting up detection probes on nvosd sink pads...")
    for i in range(number_sources):
        nvdsosd_element = pipeline.get_by_name(f"nvdsosd-{i}")
        if nvdsosd_element:
            osdsinkpad = nvdsosd_element.get_static_pad("sink")
            if osdsinkpad:
                osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, pgie_src_pad_buffer_probe, 0)
                print(f"Added detection probe to nvosd-{i} sink pad")
            else:
                sys.stderr.write(f"Unable to get sink pad of nvosd-{i}\n")
        else:
            sys.stderr.write(f"Unable to find nvosd-{i} element\n")
    
    # perf callback function to print fps every 5 sec
    GLib.timeout_add(5000, perf_data.perf_print_callback)

    # List the sources
    print("Now playing...")
    for i, source in enumerate(input_sources):
        print(i, ": ", source)

    # Add frame grabbers to capture CLEAN frames (before nvosd)
    print("Adding frame capture capabilities to pipeline...")
    for i in range(number_sources):
        # Attach frame grabber to the clean capsfilter that outputs RGBA
        # This is BEFORE nvosd, so we get clean frames without DeepStream's bounding boxes
        element_to_probe = f"capsfilter-clean-{i}"
        print(f"Setting up frame grabber for stream {i} on element {element_to_probe}")
        if not frame_grabber.add_to_pipeline(pipeline, i, element_to_probe):
            print(f"Warning: Failed to add frame capture for stream {i}")
        else:
            print(f"Successfully added frame grabber for stream {i} - should capture CLEAN frames")

    # Start the pipeline
    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)
    
    try:
        loop.run()
    except Exception as e:
        print("Caught exception:", str(e))
    finally:
        print("Exiting app\n")
        cleanup()
        # Stop pipeline
        pipeline.set_state(Gst.State.NULL)


def cleanup():
    """Clean up resources when script exits"""
    global frame_grabber, inference_saver
    
    if frame_grabber:
        try:
            frame_grabber.cleanup()
            print("Frame grabber cleaned up")
        except Exception as e:
            print(f"Error cleaning up frame grabber: {e}")
            
    if inference_saver:
        try:
            inference_saver.close()
            print("Structured saver closed")
        except Exception as e:
            print(f"Error closing structured saver: {e}")

def parse_args():
    parser = argparse.ArgumentParser(prog="deepstream_demux_multi_in_multi_out.py", 
        description="deepstream-demux-multi-in-multi-out takes multiple URI streams as input" \
            "and uses `nvstreamdemux` to split batches and output separate buffer/streams")
    parser.add_argument(
        "-i",
        "--input",
        help="Path to input streams",
        nargs="+",
        metavar="URIs",
        default=["a"],
        required=True,
    )
    parser.add_argument(
        "--bbox",
        action="store_true",
        help="Enable bounding box drawing on inferenced images (default: disabled)",
    )
    parser.add_argument(
        "--dataset",
        action="store_true",
        help="Enable dataset mode - creates YOLO format dataset with images/ and labels/ folders",
    )

    args = parser.parse_args()
    stream_paths = args.input
    global draw_bounding_boxes, dataset_mode
    draw_bounding_boxes = args.bbox
    dataset_mode = args.dataset
    return stream_paths


if __name__ == "__main__":
    stream_paths = parse_args()
    sys.exit(main(stream_paths))


