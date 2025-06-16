import os
import numpy as np
import cv2
import gi
import sys
import time
from threading import Thread, Lock
from common.platform_info import PlatformInfo

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject, GLib
import pyds

class FrameGrabber:
    """
    A robust frame grabber for DeepStream based on NVIDIA's recommended approach
    for extracting frames from NVMM memory on both Jetson and dGPU platforms.
    """
    
    def __init__(self):
        """Initialize the frame grabber"""
        self.frame_buffers = {}  # stream_id -> latest frame
        self.frame_locks = {}    # stream_id -> lock for thread safety
        self.probes = {}         # stream_id -> probe info
        self.platform_info = PlatformInfo()
        
        print(f"Frame grabber initialized on {'integrated GPU' if self.platform_info.is_integrated_gpu() else 'dGPU'} platform")
    
    def add_to_pipeline(self, pipeline, stream_id, element_name):
        """
        Add a buffer probe to capture frames from the specified element in the pipeline.
        
        Args:
            pipeline: The GStreamer pipeline
            stream_id: ID of the stream to capture
            element_name: Name of the element to tap into (should output RGBA NVMM)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Skip if already added for this stream
            if stream_id in self.probes:
                print(f"Frame grabber probe already exists for stream {stream_id}")
                return True
            
            # Initialize lock for this stream
            self.frame_locks[stream_id] = Lock()
            
            print(f"Adding frame grabber for stream {stream_id} on {element_name} (CLEAN FRAMES)")
            
            # Get the element from the pipeline
            element = pipeline.get_by_name(element_name)
            if not element:
                print(f"Could not find element {element_name}")
                return False
            
            # Get the source pad to attach the probe to
            src_pad = element.get_static_pad("src")
            if not src_pad:
                print(f"Could not get src pad from {element_name}")
                return False
            
            # Initialize frame buffer with defaults
            with self.frame_locks[stream_id]:
                self.frame_buffers[stream_id] = {
                    'frame': None,
                    'frame_num': -1,  # Track frame number for synchronization
                    'width': 640,  # Default
                    'height': 480, # Default
                    'last_update': 0
                }
            
            # Add buffer probe
            probe_id = src_pad.add_probe(
                Gst.PadProbeType.BUFFER,
                self._buffer_probe_callback,
                stream_id
            )
            
            if probe_id:
                self.probes[stream_id] = {
                    "id": probe_id,
                    "pad": src_pad,
                    "element": element_name
                }
                print(f"Successfully added CLEAN frame grabber for stream {stream_id}")
                
                # Start a monitoring thread to verify frame capture
                self._start_monitoring_thread(stream_id)
                return True
            else:
                print(f"Failed to add probe to {element_name}")
                return False
                
        except Exception as e:
            print(f"Error setting up frame grabber: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _start_monitoring_thread(self, stream_id):
        """
        Start a thread to periodically check if frames are being captured.
        """
        def monitor_frames():
            while True:
                try:
                    time.sleep(5)  # Check every 5 seconds
                    
                    frame = self.get_frame(stream_id)
                    if frame is None:
                        print(f"Monitor: No frame available for stream {stream_id}")
                        
                except Exception as e:
                    print(f"Error in frame monitor: {e}")
        
        thread = Thread(target=monitor_frames, daemon=True)
        thread.start()
    
    def _buffer_probe_callback(self, pad, info, stream_id):
        """
        Buffer probe callback that extracts frames using NVIDIA's recommended method.
        This follows the approach from the NVIDIA DeepStream Python example.
        """
        try:
            # Get the buffer
            gst_buffer = info.get_buffer()
            if not gst_buffer:
                return Gst.PadProbeReturn.OK
            
            # Get batch metadata
            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
            if not batch_meta:
                return Gst.PadProbeReturn.OK
                
            # Process each frame in the batch
            l_frame = batch_meta.frame_meta_list
            while l_frame is not None:
                try:
                    # Cast to NvDsFrameMeta
                    frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                    
                    # In demux pipeline, after nvstreamdemux, each branch processes a single stream
                    # So we should capture every frame that comes through this branch for our stream
                    print(f"[FrameGrabber] Processing frame {frame_meta.frame_num} for stream {stream_id} (pad_index: {frame_meta.pad_index})")
                    
                    # Get the surface using NVIDIA's recommended method
                    n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                    
                    if n_frame is not None:
                        # Make a copy of frame data in system memory
                        # This is critical for proper memory management
                        frame_copy = np.array(n_frame, copy=True, order='C')
                        
                        # Convert RGBA to BGR for OpenCV
                        # Note: We follow NVIDIA's example using RGBA2BGRA 
                        # and then take only the BGR channels
                        frame_bgra = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGRA)
                        frame_bgr = frame_bgra[:, :, :3]  # Extract BGR channels
                        
                        # Store the frame with frame number as key for better synchronization
                        with self.frame_locks[stream_id]:
                            self.frame_buffers[stream_id]['frame'] = frame_bgr
                            self.frame_buffers[stream_id]['frame_num'] = frame_meta.frame_num
                            self.frame_buffers[stream_id]['width'] = frame_meta.source_frame_width
                            self.frame_buffers[stream_id]['height'] = frame_meta.source_frame_height
                            self.frame_buffers[stream_id]['last_update'] = time.time()
                        
                        print(f"[FrameGrabber] Successfully captured CLEAN frame {frame_meta.frame_num} for stream {stream_id}")
                        
                        # For Jetson (integrated GPU), we must unmap the buffer when done
                        if self.platform_info.is_integrated_gpu():
                            # This is critical to prevent memory leaks on Jetson
                            pyds.unmap_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                    
                    # After demux, typically only one frame per buffer
                    break
                    
                except Exception as e:
                    print(f"[FrameGrabber] Error processing frame: {e}")
                    break
                    
                try:
                    l_frame = l_frame.next
                except StopIteration:
                    break
            
            return Gst.PadProbeReturn.OK
        
        except Exception as e:
            print(f"[FrameGrabber] Exception in buffer probe: {e}")
            return Gst.PadProbeReturn.OK
    
    def get_frame(self, stream_id):
        """
        Get the latest frame for a stream
        
        Args:
            stream_id: ID of the stream
            
        Returns:
            BGR numpy array or None if no frame is available
        """
        try:
            if stream_id in self.frame_buffers and stream_id in self.frame_locks:
                with self.frame_locks[stream_id]:
                    frame_data = self.frame_buffers[stream_id]
                    if frame_data['frame'] is not None:
                        # Return a copy to avoid threading issues
                        return frame_data['frame'].copy()
        except Exception as e:
            print(f"Error getting frame: {e}")
        return None
    
    def get_black_frame(self, stream_id):
        """
        Create a black frame with the same dimensions as the stream
        
        Args:
            stream_id: ID of the stream
            
        Returns:
            Black BGR frame with correct dimensions
        """
        try:
            if stream_id in self.frame_buffers and stream_id in self.frame_locks:
                with self.frame_locks[stream_id]:
                    frame_data = self.frame_buffers[stream_id]
                    width = frame_data.get('width', 640)
                    height = frame_data.get('height', 480)
                    return np.zeros((height, width, 3), dtype=np.uint8)
        except Exception as e:
            print(f"Error creating black frame: {e}")
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def cleanup(self):
        """Release all resources"""
        try:
            # Remove all probes
            for stream_id, probe_info in self.probes.items():
                if 'pad' in probe_info and 'id' in probe_info:
                    pad = probe_info['pad']
                    probe_id = probe_info['id']
                    pad.remove_probe(probe_id)
                    print(f"Removed probe for stream {stream_id}")
            
            # Clear all structures
            self.frame_buffers.clear()
            self.frame_locks.clear()
            self.probes.clear()
            print("Frame grabber resources released")
        except Exception as e:
            print(f"Error during cleanup: {e}")
