import os
import csv
import cv2
import numpy as np
import time
from datetime import datetime
import pyds
import threading
try:
    from jtop import jtop
    JTOP_AVAILABLE = True
except ImportError:
    print("Warning: jtop not available. System metrics monitoring will be disabled.")
    JTOP_AVAILABLE = False
    jtop = None
import json
import traceback

class DetectionMemory:
    """
    Tracks recent detection bounding boxes to avoid saving similar detections.
    Uses Intersection over Union (IoU) to measure similarity.
    """
    def __init__(self, iou_threshold=0.7, memory_duration=30):
        """
        Args:
            iou_threshold: Threshold above which detections are considered similar
            memory_duration: How long to keep detections in memory (in seconds)
        """
        self.iou_threshold = iou_threshold
        self.memory_duration = memory_duration
        self.detections = {}  # stream_id -> class_id -> list of (bbox, timestamp)
        
    def calculate_iou(self, box1, box2):
        """
        Calculate IoU between two bounding boxes.
        
        Args:
            box1: First box (left, top, width, height)
            box2: Second box (left, top, width, height)
            
        Returns:
            IoU value between 0 and 1
        """
        # Convert to (x1, y1, x2, y2) format
        x1_1, y1_1 = box1[0], box1[1]
        x2_1, y2_1 = box1[0] + box1[2], box1[1] + box1[3]
        
        x1_2, y1_2 = box2[0], box2[1]
        x2_2, y2_2 = box2[0] + box2[2], box2[1] + box2[3]
        
        # Calculate area of each box
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0  # No intersection
        
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        union_area = area1 + area2 - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0.0
        return iou

    def is_similar(self, stream_id, class_id, bbox, current_time):
        """
        Check if the detection is similar to any recent detections for the same stream and class.
        
        Args:
            stream_id: Stream identifier
            class_id: Class identifier
            bbox: Bounding box (left, top, width, height)
            current_time: Current timestamp
            
        Returns:
            True if the detection is similar to a recent one, False otherwise
        """
        # If we don't have any detections for this stream yet, it's not similar
        if stream_id not in self.detections:
            return False
            
        # If we don't have any detections for this class yet, it's not similar
        if class_id not in self.detections[stream_id]:
            return False
            
        # Check against recent detections
        for stored_bbox, timestamp in self.detections[stream_id][class_id]:
            # Skip detections that are outside the temporal window
            if current_time - timestamp > self.memory_duration:
                continue
                
            # Calculate IoU
            iou = self.calculate_iou(bbox, stored_bbox)
            
            # If IoU is above threshold, it's similar
            if iou >= self.iou_threshold:
                return True
                
        # No similar detections found
        return False

    def update(self, stream_id, class_id, bbox, current_time):
        """
        Add a new detection to memory.
        
        Args:
            stream_id: Stream identifier
            class_id: Class identifier
            bbox: Bounding box (left, top, width, height)
            current_time: Current timestamp
        """
        # Ensure we have dictionaries for this stream and class
        if stream_id not in self.detections:
            self.detections[stream_id] = {}
            
        if class_id not in self.detections[stream_id]:
            self.detections[stream_id][class_id] = []
            
        # Add the detection
        self.detections[stream_id][class_id].append((bbox, current_time))
        
        # Clean up old detections
        self.clean_old_detections(current_time)

    def clean_old_detections(self, current_time):
        """
        Remove detections that are outside the temporal window.
        
        Args:
            current_time: Current timestamp
        """
        for stream_id in list(self.detections.keys()):
            for class_id in list(self.detections[stream_id].keys()):
                # Filter out old detections
                self.detections[stream_id][class_id] = [
                    (bbox, timestamp) 
                    for bbox, timestamp in self.detections[stream_id][class_id]
                    if current_time - timestamp <= self.memory_duration
                ]
                
                # Remove empty classes
                if not self.detections[stream_id][class_id]:
                    del self.detections[stream_id][class_id]
                    
            # Remove empty streams
            if not self.detections[stream_id]:
                del self.detections[stream_id]

class SessionManager:
    """
    Manages session-based folder rotation for saving inference results.
    Creates time-stamped session folders and rotates them at specified intervals.
    """
    def __init__(self, base_dir="data", rotation_interval=15):  # 300 seconds (5 min) for production
        self.base_dir = base_dir
        self.rotation_interval = rotation_interval  # seconds
        self.current_session = None
        self.session_start_time = 0
        os.makedirs(self.base_dir, exist_ok=True)
        self.create_new_session()
        
    def create_new_session(self):
        """Create a new session folder with timestamp format MM-DD-YY-HH-MM-SS-fff"""
        timestamp = datetime.now().strftime("%m-%d-%y-%H-%M-%S-%f")[:-3]
        self.current_session = os.path.join(self.base_dir, timestamp)
        self.session_start_time = time.time()
        
        # Create session directory only
        os.makedirs(self.current_session, exist_ok=True)
        print(f"Created new session: {self.current_session}")
        return self.current_session
    
    def get_active_session(self):
        """Return the active session path, creating a new one if needed"""
        current_time = time.time()
        elapsed_time = current_time - self.session_start_time
        
        # Check if it's time to rotate
        if elapsed_time > self.rotation_interval:
            print(f"Time to rotate! Creating new session...")
            self.create_new_session()
        return self.current_session
    
    def get_active_base_dir(self, subdir=None):
        """Get the active base directory for saving files"""
        if subdir:
            return os.path.join(self.get_active_session(), subdir)
        return self.get_active_session()

class InferenceSaver:
    """
    Handles structured saving of inference results (CSV and images) for DeepStream multi-stream analytics.
    Uses session-based folder rotation with direct camera/class organization.
    """
    # All 80 COCO classes that should be saved
    CLASS_NAMES = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        4: "airplane",
        5: "bus",
        6: "train",
        7: "truck",
        8: "boat",
        9: "traffic_light",
        10: "fire_hydrant",
        11: "stop_sign",
        12: "parking_meter",
        13: "bench",
        14: "bird",
        15: "cat",
        16: "dog",
        17: "horse",
        18: "sheep",
        19: "cow",
        20: "elephant",
        21: "bear",
        22: "zebra",
        23: "giraffe",
        24: "backpack",
        25: "umbrella",
        26: "handbag",
        27: "tie",
        28: "suitcase",
        29: "frisbee",
        30: "skis",
        31: "snowboard",
        32: "sports_ball",
        33: "kite",
        34: "baseball_bat",
        35: "baseball_glove",
        36: "skateboard",
        37: "surfboard",
        38: "tennis_racket",
        39: "bottle",
        40: "wine_glass",
        41: "cup",
        42: "fork",
        43: "knife",
        44: "spoon",
        45: "bowl",
        46: "banana",
        47: "apple",
        48: "sandwich",
        49: "orange",
        50: "broccoli",
        51: "carrot",
        52: "hot_dog",
        53: "pizza",
        54: "donut",
        55: "cake",
        56: "chair",
        57: "couch",
        58: "potted_plant",
        59: "bed",
        60: "dining_table",
        61: "toilet",
        62: "tv",
        63: "laptop",
        64: "mouse",
        65: "remote",
        66: "keyboard",
        67: "cell_phone",
        68: "microwave",
        69: "oven",
        70: "toaster",
        71: "sink",
        72: "refrigerator",
        73: "book",
        74: "clock",
        75: "vase",
        76: "scissors",
        77: "teddy_bear",
        78: "hair_drier",
        79: "toothbrush"
    }
    
    # All classes to save (all 80 COCO classes)
    SAVE_CLASSES = CLASS_NAMES.copy()

    def __init__(self, base_dir, enable_bbox_similarity_filtering=False, iou_threshold=0.7, memory_seconds=5, enable_system_metrics=True, metrics_interval=60, draw_bboxes=False, dataset_mode=False):
        self.base_dir = base_dir
        self.session_manager = SessionManager(base_dir=base_dir)
        self.camera_names = {}   # stream_id -> camera name
        self.csv_files = {}      # (camera_name, class_name) -> file handle
        self._frame_extraction_method = 0
        self.camera_mapping = {}
        self.draw_bboxes = draw_bboxes  # Store bbox drawing flag
        self.dataset_mode = dataset_mode  # Store dataset mode flag
        
        # Generate colors for all 80 classes using HSV color space for better distribution
        self.bbox_colors = self._generate_class_colors()
        
        # Initialize detection memory if similarity filtering is enabled
        if enable_bbox_similarity_filtering:
            self.detection_memory = DetectionMemory(
                iou_threshold=iou_threshold,
                memory_duration=memory_seconds
            )
            print(f"Bounding box similarity filtering enabled (IoU threshold: {iou_threshold}, memory: {memory_seconds}s)")
        else:
            self.detection_memory = None
        
        # System metrics monitoring (only if jtop is available)
        self.enable_system_metrics = enable_system_metrics and JTOP_AVAILABLE
        self.metrics_interval = metrics_interval  # seconds between metric logs
        self.system_metrics_csv = None
        self.system_metrics_thread = None
        self.system_metrics_stop_event = threading.Event()
        
        # Add dedicated session rotation timer
        self.session_rotation_stop_event = threading.Event()
        self.session_rotation_thread = threading.Thread(target=self._session_rotation_timer, daemon=True)
        self.session_rotation_thread.start()
        
        print(f"Session rotation timer started - will rotate every {self.session_manager.rotation_interval} seconds")
        print(f"Bounding box drawing: {'ENABLED' if self.draw_bboxes else 'DISABLED'}")
        print(f"Dataset mode: {'ENABLED' if self.dataset_mode else 'DISABLED'}")

        if enable_system_metrics and not JTOP_AVAILABLE:
            print("Warning: System metrics requested but jtop not available. Metrics disabled.")
        
        # Start system metrics logging if enabled and available
        if self.enable_system_metrics:
            self.setup_system_metrics_logging()
            
        print(f"InferenceSaver initialized with base directory: {base_dir}")
        if self.enable_system_metrics:
            print(f"System metrics logging interval: {metrics_interval} seconds")
        else:
            print("System metrics logging disabled")
    
    def _generate_class_colors(self):
        """
        Generate distinct colors for all 80 COCO classes using HSV color space.
        Returns a dictionary mapping class_id to BGR color tuple.
        """
        import colorsys
        
        colors = {}
        num_classes = len(self.CLASS_NAMES)
        
        for class_id in range(num_classes):
            # Generate hue values distributed across the color spectrum
            hue = (class_id * 137.508) % 360  # Use golden angle for good distribution
            saturation = 0.8 + (class_id % 3) * 0.1  # Vary saturation slightly
            value = 0.8 + (class_id % 2) * 0.2  # Vary brightness slightly
            
            # Convert HSV to RGB
            rgb = colorsys.hsv_to_rgb(hue/360.0, saturation, value)
            
            # Convert to BGR (OpenCV format) and scale to 0-255
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            colors[class_id] = bgr
            
        return colors

    def _session_rotation_timer(self):
        """
        Dedicated background thread that handles session rotation based purely on time.
        This ensures sessions rotate regardless of detection activity.
        """
        try:
            print(f"Session rotation timer started with {self.session_manager.rotation_interval}s interval")
            
            while not self.session_rotation_stop_event.wait(5):  # Check every 5 seconds
                try:
                    # Check if it's time to rotate
                    current_time = time.time()
                    elapsed_time = current_time - self.session_manager.session_start_time
                    
                    if elapsed_time >= self.session_manager.rotation_interval:
                        print(f"Time-based rotation triggered: {elapsed_time:.1f}s >= {self.session_manager.rotation_interval}s")
                        self._perform_session_rotation()
                        
                except Exception as e:
                    print(f"Error in session rotation timer: {e}")
                    traceback.print_exc()
                    
        except Exception as e:
            print(f"Session rotation timer thread failed: {e}")
            traceback.print_exc()    
            
    def _perform_session_rotation(self):
        """
        Perform the actual session rotation - create new session and rotate all files.
        This is called by the timer thread when it's time to rotate.
        """
        try:
            # Get old session for logging
            old_session = self.session_manager.current_session
            
            # Force create new session
            new_session = self.session_manager.create_new_session()
            
            print(f"TIMER ROTATION: {os.path.basename(old_session)} -> {os.path.basename(new_session)}")
            
            # Close all CSV files
            for file_info in self.csv_files.values():
                try:
                    file_info["file"].close()
                except:
                    pass
            
            # Rotate system metrics CSV if enabled
            if self.enable_system_metrics and self.system_metrics_csv:
                try:
                    print("Rotating system metrics CSV for new session")
                    self.system_metrics_csv["file"].close()
                    
                    # Setup new system metrics CSV file in session root
                    session_base = self.get_active_base_dir()
                    csv_path = os.path.join(session_base, "system_metrics.csv")
                    csvfile = open(csv_path, 'w', newline='')
                    fieldnames = [
                        'timestamp', 'date', 'time',
                        'cpu_usage_percent', 'gpu_usage_percent', 'memory_used_mb', 'memory_total_mb', 'memory_usage_percent',
                        'swap_used_mb', 'swap_total_mb', 'swap_usage_percent',
                        'disk_used_gb', 'disk_total_gb', 'disk_usage_percent',
                        'cpu_temp_celsius', 'gpu_temp_celsius', 'thermal_temp_celsius',
                        'power_total_watts', 'power_cpu_watts', 'power_gpu_watts',
                        'fan_speed_percent', 'uptime_seconds'
                    ]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    self.system_metrics_csv = {
                        "file": csvfile,
                        "writer": writer
                    }
                    
                    print(f"System metrics CSV rotated to: {csv_path}")
                    
                except Exception as e:
                    print(f"Error rotating system metrics CSV: {e}")
            
            # Clear CSV files dictionary to force recreation on next detection
            self.csv_files.clear()
            
            # Only re-setup basic camera directories (not class directories)
            for stream_id, camera_name in self.camera_names.items():
                self.setup_camera_structure(camera_name)
            
            print("Session rotation complete - all files rotated to new session")
            
        except Exception as e:
            print(f"Error during session rotation: {e}")
            traceback.print_exc()
    
    def setup_system_metrics_logging(self):
        """Setup system metrics CSV file and start monitoring thread"""
        if not JTOP_AVAILABLE:
            print("Warning: Cannot setup system metrics - jtop not available")
            return
            
        try:
            # Save CSV directly in session root
            session_base = self.get_active_base_dir()
            
            # Setup CSV file directly in session root
            csv_path = os.path.join(session_base, "system_metrics.csv")
            file_exists = os.path.isfile(csv_path)
            
            csvfile = open(csv_path, 'a', newline='')
            fieldnames = [
                'timestamp', 'date', 'time',
                'cpu_usage_percent', 'gpu_usage_percent', 'memory_used_mb', 'memory_total_mb', 'memory_usage_percent',
                'swap_used_mb', 'swap_total_mb', 'swap_usage_percent',
                'disk_used_gb', 'disk_total_gb', 'disk_usage_percent',
                'cpu_temp_celsius', 'gpu_temp_celsius', 'thermal_temp_celsius',
                'power_total_watts', 'power_cpu_watts', 'power_gpu_watts',
                'fan_speed_percent', 'uptime_seconds'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
                
            self.system_metrics_csv = {
                "file": csvfile,
                "writer": writer
            }
            
            # Start monitoring thread
            self.system_metrics_thread = threading.Thread(target=self._monitor_system_metrics, daemon=True)
            self.system_metrics_thread.start()
            
            print(f"System metrics logging started: {csv_path}")
            
        except Exception as e:
            print(f"Error setting up system metrics logging: {e}")
            self.enable_system_metrics = False
            
    def _monitor_system_metrics(self):
        """Background thread function to monitor and log system metrics"""
        if not JTOP_AVAILABLE:
            return
        try:
            print(f"=== SYSTEM METRICS DEBUG ===")
            print(f"Metrics interval: {self.metrics_interval} seconds")
            print(f"Starting monitoring thread...")
            
            with jtop() as jetson:
                while not self.system_metrics_stop_event.wait(self.metrics_interval):
                    try:
                        # REMOVE THIS LINE: self._rotate_system_metrics_if_needed()
                        
                        if not self.system_metrics_csv:
                            continue
                            
                        # Collect system metrics
                        now = datetime.now()
                        
                        # CPU usage - get total CPU usage from cpu property
                        cpu_data = jetson.cpu
                        cpu_total = cpu_data.get('total', {})
                        cpu_usage_percent = cpu_total.get('user', 0) + cpu_total.get('system', 0)
                        
                        # GPU usage - from stats or gpu property
                        gpu_usage_percent = jetson.stats.get('GPU', 0)
                        
                        # Memory stats - from memory property
                        memory_data = jetson.memory
                        ram_data = memory_data.get('RAM', {})
                        memory_used_kb = ram_data.get('used', 0)
                        memory_total_kb = ram_data.get('tot', 1)
                        memory_used_mb = memory_used_kb / 1024
                        memory_total_mb = memory_total_kb / 1024
                        memory_usage_percent = (memory_used_kb / max(memory_total_kb, 1)) * 100
                        
                        # SWAP stats
                        swap_data = memory_data.get('SWAP', {})
                        swap_used_kb = swap_data.get('used', 0)
                        swap_total_kb = swap_data.get('tot', 1)
                        swap_used_mb = swap_used_kb / 1024
                        swap_total_mb = swap_total_kb / 1024
                        swap_usage_percent = (swap_used_kb / max(swap_total_kb, 1)) * 100
                        
                        # Disk stats - from disk property
                        disk_data = jetson.disk
                        disk_used_gb = disk_data.get('used', 0)
                        disk_total_gb = disk_data.get('total', 1)
                        disk_usage_percent = (disk_used_gb / max(disk_total_gb, 1)) * 100
                        
                        # Temperature stats - from temperature property
                        temp_data = jetson.temperature
                        cpu_temp_celsius = temp_data.get('cpu', {}).get('temp', 0)
                        # GPU temp might be -256 when offline, so handle it
                        gpu_temp_raw = temp_data.get('gpu', {}).get('temp', 0)
                        gpu_temp_celsius = gpu_temp_raw if gpu_temp_raw > -200 else 0
                        # Use tj (junction) temperature as thermal temp
                        thermal_temp_celsius = temp_data.get('tj', {}).get('temp', 0)
                        
                        # Power stats - from power property
                        power_data = jetson.power
                        power_tot = power_data.get('tot', {})
                        power_total_watts = power_tot.get('power', 0) / 1000.0  # Convert mW to W
                        
                        # Individual rail powers (convert mW to W)
                        power_rails = power_data.get('rail', {})
                        power_cpu_watts = power_rails.get('VDD_CPU_CV', {}).get('power', 0) / 1000.0
                        power_gpu_watts = power_rails.get('VDD_GPU_SOC', {}).get('power', 0) / 1000.0
                        
                        # Fan stats - from fan property or stats
                        fan_data = jetson.fan
                        fan_speed_percent = 0
                        if hasattr(fan_data, 'get') and 'pwmfan' in fan_data:
                            fan_speeds = fan_data['pwmfan'].get('speed', [0])
                            fan_speed_percent = fan_speeds[0] if fan_speeds else 0
                        else:
                            # Fallback to stats
                            fan_speed_percent = jetson.stats.get('Fan pwmfan0', 0)
                        
                        # Uptime
                        uptime_delta = jetson.uptime
                        uptime_seconds = uptime_delta.total_seconds() if uptime_delta else 0
                        
                        metrics_data = {
                            'timestamp': now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                            'date': now.strftime("%Y-%m-%d"),
                            'time': now.strftime("%H:%M:%S.%f")[:-3],
                            
                            # CPU metrics
                            'cpu_usage_percent': round(cpu_usage_percent, 2),
                            
                            # GPU metrics
                            'gpu_usage_percent': round(gpu_usage_percent, 2),
                            
                            # Memory metrics (convert KB to MB)
                            'memory_used_mb': round(memory_used_mb, 2),
                            'memory_total_mb': round(memory_total_mb, 2),
                            'memory_usage_percent': round(memory_usage_percent, 2),
                            
                            # Swap metrics (convert KB to MB)
                            'swap_used_mb': round(swap_used_mb, 2),
                            'swap_total_mb': round(swap_total_mb, 2),
                            'swap_usage_percent': round(swap_usage_percent, 2),
                            
                            # Disk metrics
                            'disk_used_gb': round(disk_used_gb, 2),
                            'disk_total_gb': round(disk_total_gb, 2),
                            'disk_usage_percent': round(disk_usage_percent, 2),
                            
                            # Temperature metrics
                            'cpu_temp_celsius': round(cpu_temp_celsius, 2),
                            'gpu_temp_celsius': round(gpu_temp_celsius, 2),
                            'thermal_temp_celsius': round(thermal_temp_celsius, 2),
                            
                            # Power metrics (convert mW to W)
                            'power_total_watts': round(power_total_watts, 2),
                            'power_cpu_watts': round(power_cpu_watts, 2),
                            'power_gpu_watts': round(power_gpu_watts, 2),
                            
                            # Fan metrics
                            'fan_speed_percent': round(fan_speed_percent, 2),
                            
                            # Uptime
                            'uptime_seconds': round(uptime_seconds, 2)
                        }
                        
                        # Write to CSV
                        self.system_metrics_csv["writer"].writerow(metrics_data)
                        self.system_metrics_csv["file"].flush()  # Ensure data is written
                        
                        print(f"Logged system metrics - CPU: {metrics_data['cpu_usage_percent']:.1f}%, "
                            f"GPU: {metrics_data['gpu_usage_percent']:.1f}%, "
                            f"RAM: {metrics_data['memory_usage_percent']:.1f}%, "
                            f"Power: {metrics_data['power_total_watts']:.1f}W")
                        
                    except Exception as e:
                        print(f"Error collecting system metrics: {e}")
                        traceback.print_exc()
                        time.sleep(1)  # Wait before retrying
                        
        except Exception as e:
            print(f"Error in system metrics monitoring thread: {e}")
            traceback.print_exc()

                        
    def get_active_base_dir(self, subdir=None):
        """Get the active base directory for the current session"""
        return self.session_manager.get_active_base_dir(subdir)

    def map_source_to_camera_name(self, stream_id, source_uri=None):
        """
        Maps stream ID to a simple camera folder name.
        Uses cam{stream_id} format for all cameras regardless of source type.
        """
        if stream_id in self.camera_names:
            return self.camera_names[stream_id]
            
        # Simple naming: cam0, cam1, cam2, etc.
        cam_name = f"cam{stream_id}"
        self.camera_names[stream_id] = cam_name
        print(f"Mapped stream {stream_id} to camera name '{cam_name}'")
        return cam_name

    def setup_camera_structure(self, camera_name):
        """Set up camera directory structure in the active session."""
        # In dataset mode, we don't need camera directories - only images/ and labels/
        if self.dataset_mode:
            return
            
        session_base = self.get_active_base_dir()
        
        # Create only the camera directory - class directories will be created on demand
        camera_dir = os.path.join(session_base, camera_name)
        os.makedirs(camera_dir, exist_ok=True)
        print(f"Set up camera directory: {camera_dir}")

    def setup_stream(self, stream_id, source_uri=None):
        """
        Set up directories for a stream in the active session with camera/class organization.
        """
        # Get the camera name for this stream
        camera_name = self.map_source_to_camera_name(stream_id, source_uri)
        
        # Only setup camera structure if not in dataset mode
        if not self.dataset_mode:
            self.setup_camera_structure(camera_name)
        
        print(f"Set up stream {stream_id} as camera '{camera_name}' in current session")
        return camera_name

    def extract_frame_from_buffer(self, buffer, frame_meta):
        """
        Extracts a BGR numpy array from the GStreamer buffer for the specified frame.
        Uses multiple methods for maximum compatibility with different DeepStream versions.
        
        Args:
            buffer: GStreamer buffer containing the frame data
            frame_meta: NvDsFrameMeta object containing metadata for the frame
            
        Returns:
            BGR numpy array or None if extraction fails
        """

        
        # Method 1: Using pyds.get_nvds_buf_surface (DeepStream 5.0+)
        try:
            # This method works for RGBA surfaces in DeepStream 5.0 and 6.0
            n_frame = pyds.get_nvds_buf_surface(hash(buffer), frame_meta.batch_id)
            if n_frame is not None:
                print(f"Surface extracted, shape: {n_frame.shape}, dtype: {n_frame.dtype}")
                # If we got a valid frame, process it based on format
                if len(n_frame.shape) == 3:  # Has shape (height, width, channels)
                    if n_frame.shape[2] == 4:  # RGBA
                        self._frame_extraction_method = 1
                        return cv2.cvtColor(np.array(n_frame, copy=True, order='C'), cv2.COLOR_RGBA2BGR)
                    elif n_frame.shape[2] == 3:  # RGB
                        self._frame_extraction_method = 1
                        return cv2.cvtColor(np.array(n_frame, copy=True, order='C'), cv2.COLOR_RGB2BGR)
                    else:
                        print(f"Unsupported channel count: {n_frame.shape[2]}")
                else:
                    print(f"Unexpected frame shape: {n_frame.shape}")
        except Exception as e:
            print(f"Method 1 extraction failed: {e}")
        
        # Method 2: Using NvDsFrameMeta and NvBufSurface (DeepStream 6.0+)
        try:
            # Try to access the surface directly from the frame_meta
            if hasattr(frame_meta, 'surface') and frame_meta.surface:
                print("Trying Method 2: direct access to frame_meta.surface")
                surface = frame_meta.surface
                # Convert the surface to a numpy array (implementation depends on DeepStream version)
                # This is just a placeholder - actual implementation would be platform-specific
                print("Method 2 not implemented for this DeepStream version")
        except Exception as e:
            print(f"Method 2 extraction failed: {e}")
        
        # Method 3: Use NvDsPreProcessBatchMeta to access the converted RGBA surface
        try:
            print("Trying Method 3: NvDsPreProcessBatchMeta")
            # Check if NvDsPreProcessBatchMeta is available on the buffer
            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
            if batch_meta and batch_meta.batch_user_meta_list:
                user_meta = batch_meta.batch_user_meta_list
                while user_meta:
                    try:
                        user_meta_data = pyds.NvDsUserMeta.cast(user_meta.data)
                        if user_meta_data.base_meta.meta_type == pyds.NVDS_PREPROCESS_BATCH_META:
                            preprocess_meta = pyds.NvDsPreProcessBatchMeta.cast(user_meta_data.user_meta_data)
                            # Access the converted RGBA surface
                            print("Found NvDsPreProcessBatchMeta, but implementation is platform-specific")
                    except Exception as e:
                        print(f"Error processing user meta: {e}")
                    try:
                        user_meta = user_meta.next
                    except StopIteration:
                        break
        except Exception as e:
            print(f"Method 3 extraction failed: {e}")
        
        # Method 4: Add a fallback tee element in the pipeline
        # This would involve modifying the GStreamer pipeline to add a tee element
        # after nvvideoconvert and before nvosd, with a branch going to appsink
        # This is a complex solution requiring pipeline changes
        
        print("All frame extraction methods failed. Please modify pipeline for frame extraction.")
        return None

    def save_detection(self, detection_data, stream_id, class_id, frame_num, obj_id,
                    original_frame, inferenced_frame, source_uri=None):
        """
        Save detection data. In dataset mode, saves to YOLO format structure.
        In normal mode, uses the existing camera/class structure.
        """
        # Skip if class is not in our save list
        if class_id not in self.SAVE_CLASSES:
            return
            
        # Get camera name
        camera_name = self.map_source_to_camera_name(stream_id, source_uri)
        
        try:
            # Make sure the stream is set up for the current session
            if stream_id not in self.camera_names:
                self.setup_stream(stream_id, source_uri)
            
            # Prepare detection data
            if 'timestamp' not in detection_data:
                now = datetime.now()
                detection_data['timestamp'] = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                detection_data['date'] = now.strftime("%Y-%m-%d")
                detection_data['time'] = now.strftime("%H:%M:%S.%f")[:-3]
            
            # Add standard fields
            detection_data['stream_id'] = stream_id
            detection_data['frame_num'] = frame_num
            detection_data['obj_id'] = obj_id
            detection_data['class_id'] = class_id
            detection_data['class_name'] = self.CLASS_NAMES.get(class_id, f"unknown_{class_id}")
            
            # Extract bounding box coordinates for similarity check
            bbox = (
                detection_data.get('left', 0),
                detection_data.get('top', 0),
                detection_data.get('width', 0),
                detection_data.get('height', 0)
            )
            
            # Check if this detection is similar to a recent one
            current_time = time.time()
            is_similar = False
            if self.detection_memory:
                is_similar = self.detection_memory.is_similar(stream_id, class_id, bbox, current_time)
                self.detection_memory.update(stream_id, class_id, bbox, current_time)
            
            # Get class name for saving
            class_name = self.SAVE_CLASSES[class_id]
            detection_data['class_name'] = class_name
            
            # Use clean frame for dataset mode (no bboxes in dataset images)
            frame_to_save = original_frame if self.dataset_mode else inferenced_frame
            save_type = "dataset" if self.dataset_mode else "inferenced"
            
            if self.dataset_mode:
                # Save in dataset format
                self._save_dataset_detection(detection_data, frame_to_save, save_image=not is_similar)
            else:
                # Save in existing camera/class structure
                self._save_detection(detection_data, camera_name, class_name, frame_to_save, save_image=not is_similar, save_type=save_type)
            
            if is_similar:
                print(f"Skipped saving image for similar detection (stream: {stream_id}, class: {class_id}, obj: {obj_id})")
            
        except Exception as e:
            print(f"Error in save_detection: {e}")
            import traceback
            traceback.print_exc()

    def _save_dataset_detection(self, detection_data, frame, save_image=True):
        """
        Save detection in YOLO dataset format with images/ and labels/ folders.
        Creates one image per frame with corresponding YOLO format label file.
        """
        if not save_image:
            return
            
        try:
            # Get current session directory
            session_base = self.get_active_base_dir()
            
            # Create dataset structure
            images_dir = os.path.join(session_base, "images")
            labels_dir = os.path.join(session_base, "labels")
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)
            
            # Generate filename based on timestamp and frame info
            timestamp_str = detection_data['timestamp'].replace(':', '-').replace(' ', '_')
            stream_id = detection_data['stream_id']
            frame_num = detection_data['frame_num']
            base_filename = f"{timestamp_str}_s{stream_id}_f{frame_num}"
            
            image_filename = f"{base_filename}.jpg"
            label_filename = f"{base_filename}.txt"
            
            image_path = os.path.join(images_dir, image_filename)
            label_path = os.path.join(labels_dir, label_filename)
            
            # Check if we already saved this frame (same timestamp + stream + frame_num)
            if os.path.exists(image_path):
                # Image already exists, just append to label file
                self._append_yolo_annotation(label_path, detection_data, frame)
                print(f"Appended annotation to existing label: {label_filename}")
            else:
                # New frame - save image and create label file
                if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
                    success = cv2.imwrite(image_path, frame)
                    if success:
                        # Create new label file with this detection
                        self._create_yolo_annotation(label_path, detection_data, frame)
                        print(f"Saved dataset image: {image_filename} with label: {label_filename}")
                    else:
                        print(f"Failed to save dataset image: {image_path}")
                        
        except Exception as e:
            print(f"Error in _save_dataset_detection: {e}")
            import traceback
            traceback.print_exc()

    def _create_yolo_annotation(self, label_path, detection_data, frame):
        """Create a new YOLO format annotation file."""
        try:
            with open(label_path, 'w') as f:
                yolo_line = self._detection_to_yolo_format(detection_data, frame)
                f.write(yolo_line + '\n')
        except Exception as e:
            print(f"Error creating YOLO annotation: {e}")

    def _append_yolo_annotation(self, label_path, detection_data, frame):
        """Append to existing YOLO format annotation file."""
        try:
            with open(label_path, 'a') as f:
                yolo_line = self._detection_to_yolo_format(detection_data, frame)
                f.write(yolo_line + '\n')
        except Exception as e:
            print(f"Error appending YOLO annotation: {e}")

    def _detection_to_yolo_format(self, detection_data, frame):
        """
        Convert detection to YOLO format: class_id x_center y_center width height
        All coordinates are normalized to [0, 1]
        """
        if frame is None:
            return ""
            
        frame_height, frame_width = frame.shape[:2]
        
        # Get detection coordinates
        left = detection_data['left']
        top = detection_data['top']
        width = detection_data['width']
        height = detection_data['height']
        class_id = detection_data['class_id']
        
        # Convert to YOLO format (normalized center coordinates)
        x_center = (left + width / 2) / frame_width
        y_center = (top + height / 2) / frame_height
        norm_width = width / frame_width
        norm_height = height / frame_height
        
        # YOLO format: class_id x_center y_center width height
        return f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"

    def _save_detection(self, detection_data, camera_name, class_name, frame, save_image=True, save_type="inferenced"):
        """
        Save detection to camera/class structure in session folder.
        Creates class directories and CSV files on-demand when detections occur.
        Draws bounding box if bbox drawing is enabled.
        
        Args:
            detection_data: Dict with detection metadata
            camera_name: Camera identifier
            class_name: Class name
            frame: Frame to save
            save_image: Whether to save the image (False for similar detections)
            save_type: Type of frame being saved (for logging)
        """
        try:
            # Get current session directory
            session_base = self.get_active_base_dir()
            
            # Define paths - create directories on demand
            class_dir = os.path.join(session_base, camera_name, class_name)
            images_dir = os.path.join(class_dir, "images")
            
            # Create directories only when we have a detection to save
            os.makedirs(class_dir, exist_ok=True)
            os.makedirs(images_dir, exist_ok=True)
            
            # Set up CSV file on demand if it doesn't exist
            csv_key = (camera_name, class_name)
            if csv_key not in self.csv_files or self.csv_files[csv_key]["file"].closed:
                csv_path = os.path.join(class_dir, "metadata.csv")
                file_exists = os.path.isfile(csv_path)
                
                try:
                    csvfile = open(csv_path, 'a', newline='')
                    fieldnames = [
                        'timestamp', 'date', 'time', 'stream_id', 'frame_num', 
                        'obj_id', 'class_id', 'class_name', 'left', 'top', 'width', 
                        'height', 'confidence'
                    ]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    if not file_exists:
                        writer.writeheader()
                        
                    self.csv_files[csv_key] = {
                        "file": csvfile, 
                        "writer": writer
                    }
                    print(f"Created CSV on-demand for {camera_name}/{class_name}: {csv_path}")
                except Exception as e:
                    print(f"Error setting up CSV for {camera_name}/{class_name}: {e}")
                    return
            
            # Only save to CSV if we're also saving the image
            if save_image:
                # Save to CSV
                csv_info = self.csv_files.get(csv_key)
                if csv_info:
                    csv_info["writer"].writerow(detection_data)
                    csv_info["file"].flush()
                    print(f"Saved detection to CSV for {camera_name}/{class_name}")
                
                # Draw bounding box if enabled
                frame_to_save = frame
                if self.draw_bboxes and frame is not None:
                    frame_to_save = self.draw_bounding_box_on_frame(frame.copy(), detection_data)
                    save_type = "with_bbox"
                
                # Generate filename based on timestamp and detection info
                timestamp_str = detection_data['timestamp'].replace(':', '-').replace(' ', '_')
                frame_num = detection_data['frame_num']
                obj_id = detection_data['obj_id']
                image_filename = f"{timestamp_str}_{frame_num}_{obj_id}.jpg"
                image_path = os.path.join(images_dir, image_filename)
                
                # Save frame
                if frame_to_save is not None and isinstance(frame_to_save, np.ndarray) and frame_to_save.size > 0:
                    try:
                        success = cv2.imwrite(image_path, frame_to_save)
                        if success:
                            print(f"Saved {save_type} {class_name} frame to {camera_name}/{class_name}: {image_path}")
                        else:
                            print(f"Failed to save image to {image_path}")
                    except Exception as e:
                        print(f"Error saving frame: {e}")
                                    
        except Exception as e:
            print(f"Error in _save_detection: {e}")
            import traceback
            traceback.print_exc()

    def draw_bounding_box_on_frame(self, frame, detection_data):
        """
        Draw a single bounding box on the frame for a specific detection.
        
        Args:
            frame: Frame to draw on
            detection_data: Detection data containing bbox coordinates and class info
            
        Returns:
            Frame with bounding box drawn
        """
        if not self.draw_bboxes or frame is None:
            return frame
            
        try:
            class_id = detection_data['class_id']
            left = int(detection_data['left'])
            top = int(detection_data['top'])
            width = int(detection_data['width'])
            height = int(detection_data['height'])
            class_name = detection_data['class_name']
            confidence = detection_data.get('confidence', 0.0)
            
            # Get color for this class
            color = self.bbox_colors.get(class_id, (255, 255, 255))  # Default white
            
            # Draw rectangle
            cv2.rectangle(frame, (left, top), (left + width, top + height), color, 2)
            
            # Add label with class name and confidence
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # Draw label background
            cv2.rectangle(frame, (left, top - label_size[1] - 10), 
                         (left + label_size[0], top), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (left, top - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            print(f"Drew bbox for {class_name} at ({left}, {top}, {width}, {height})")
            
        except Exception as e:
            print(f"Error drawing bounding box: {e}")
            
        return frame

    def draw_multiple_bboxes_on_frame(self, frame, frame_detections):
        """
        Draw multiple bounding boxes on a frame for combined detections.
        
        Args:
            frame: Frame to draw on
            frame_detections: List of detection info dicts
            
        Returns:
            Frame with all bounding boxes drawn
        """
        if not self.draw_bboxes or frame is None or not frame_detections:
            return frame
            
        try:
            frame_with_bboxes = frame.copy()
            
            for detection_info in frame_detections:
                detection_data = detection_info['detection_data']
                frame_with_bboxes = self.draw_bounding_box_on_frame(frame_with_bboxes, detection_data)
                
            print(f"Drew {len(frame_detections)} bounding boxes on combined frame")
            return frame_with_bboxes
            
        except Exception as e:
            print(f"Error drawing multiple bounding boxes: {e}")
            return frame

    def save_frame_detections(self, frame_detections, camera_name, frame_to_save, save_type="inferenced"):
        """
        In dataset mode, this handles multiple detections per frame.
        In normal mode, uses existing behavior.
        """
        if not frame_detections:
            return
            
        if self.dataset_mode:
            # In dataset mode, save one image with all annotations
            self._save_dataset_frame_detections(frame_detections, frame_to_save)
        else:
            # Use existing behavior for normal mode
            try:
                session_base = self.get_active_base_dir()
                
                # Create detections directory structure
                detections_dir = os.path.join(session_base, camera_name, "detections")
                images_dir = os.path.join(detections_dir, "images")
                os.makedirs(detections_dir, exist_ok=True)
                os.makedirs(images_dir, exist_ok=True)
                
                # Set up combined detections CSV file
                csv_key = (camera_name, "detections")
                if csv_key not in self.csv_files or self.csv_files[csv_key]["file"].closed:
                    csv_path = os.path.join(detections_dir, "all_detections.csv")
                    file_exists = os.path.isfile(csv_path)
                    
                    try:
                        csvfile = open(csv_path, 'a', newline='')
                        fieldnames = [
                            'timestamp', 'date', 'time', 'stream_id', 'frame_num',
                            'total_objects', 'detected_classes', 'class_counts',
                            'all_objects_data'
                        ]
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        
                        if not file_exists:
                            writer.writeheader()
                            
                        self.csv_files[csv_key] = {
                            "file": csvfile, 
                            "writer": writer
                        }
                        print(f"Created detections CSV on-demand for {camera_name}: {csv_path}")
                    except Exception as e:
                        print(f"Error setting up detections CSV for {camera_name}: {e}")
                        return
                
                # Prepare combined metadata for this frame
                first_detection = frame_detections[0]['detection_data']
                
                # Count objects by class
                class_counts = {}
                detected_classes = set()
                all_objects_data = []
                
                for detection_info in frame_detections:
                    detection_data = detection_info['detection_data']
                    class_name = detection_data['class_name']
                    
                    detected_classes.add(class_name)
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    
                    # Store individual object data
                    obj_data = {
                        'obj_id': detection_data['obj_id'],
                        'class_id': detection_data['class_id'],
                        'class_name': class_name,
                        'left': detection_data['left'],
                        'top': detection_data['top'],
                        'width': detection_data['width'],
                        'height': detection_data['height'],
                        'confidence': detection_data['confidence']
                    }
                    all_objects_data.append(obj_data)
                
                # Create combined metadata entry
                combined_data = {
                    'timestamp': first_detection['timestamp'],
                    'date': first_detection['date'],
                    'time': first_detection['time'],
                    'stream_id': first_detection['stream_id'],
                    'frame_num': first_detection['frame_num'],
                    'total_objects': len(frame_detections),
                    'detected_classes': ', '.join(sorted(detected_classes)),
                    'class_counts': str(class_counts),
                    'all_objects_data': str(all_objects_data)
                }
                
                # Save to CSV
                csv_info = self.csv_files.get(csv_key)
                if csv_info:
                    csv_info["writer"].writerow(combined_data)
                    csv_info["file"].flush()
                    print(f"Saved combined detection data for {camera_name}/detections")
                
                # Draw all bounding boxes if enabled
                final_frame = frame_to_save
                if self.draw_bboxes and frame_to_save is not None:
                    final_frame = self.draw_multiple_bboxes_on_frame(frame_to_save.copy(), frame_detections)
                    save_type = "with_all_bboxes"
                
                # Generate filename for the frame image
                timestamp_str = first_detection['timestamp'].replace(':', '-').replace(' ', '_')
                frame_num = first_detection['frame_num']
                classes_str = '_'.join(sorted(detected_classes))
                image_filename = f"{timestamp_str}_{frame_num}_{classes_str}.jpg"
                image_path = os.path.join(images_dir, image_filename)
                
                # Save the frame with all detections
                if final_frame is not None and isinstance(final_frame, np.ndarray) and final_frame.size > 0:
                    try:
                        success = cv2.imwrite(image_path, final_frame)
                        if success:
                            print(f"Saved {save_type} combined frame to {camera_name}/detections: {image_path}")
                            print(f"Frame contains {len(frame_detections)} objects: {', '.join(sorted(detected_classes))}")
                        else:
                            print(f"Failed to save combined image to {image_path}")
                    except Exception as e:
                        print(f"Error saving combined frame: {e}")
                        
            except Exception as e:
                print(f"Error in save_frame_detections: {e}")
                import traceback
                traceback.print_exc()

    def _save_dataset_frame_detections(self, frame_detections, frame_to_save):
        """
        Save all detections from a frame to dataset format.
        Creates one image with one label file containing all annotations.
        """
        if not frame_detections:
            return
            
        try:
            # Get current session directory
            session_base = self.get_active_base_dir()
            
            # Create dataset structure
            images_dir = os.path.join(session_base, "images")
            labels_dir = os.path.join(session_base, "labels")
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)
            
            # Use first detection for filename
            first_detection = frame_detections[0]['detection_data']
            timestamp_str = first_detection['timestamp'].replace(':', '-').replace(' ', '_')
            stream_id = first_detection['stream_id']
            frame_num = first_detection['frame_num']
            base_filename = f"{timestamp_str}_s{stream_id}_f{frame_num}"
            
            image_filename = f"{base_filename}.jpg"
            label_filename = f"{base_filename}.txt"
            
            image_path = os.path.join(images_dir, image_filename)
            label_path = os.path.join(labels_dir, label_filename)
            
            # Save image
            if frame_to_save is not None and isinstance(frame_to_save, np.ndarray) and frame_to_save.size > 0:
                success = cv2.imwrite(image_path, frame_to_save)
                if success:
                    # Create label file with all detections
                    with open(label_path, 'w') as f:
                        for detection_info in frame_detections:
                            detection_data = detection_info['detection_data']
                            yolo_line = self._detection_to_yolo_format(detection_data, frame_to_save)
                            if yolo_line:
                                f.write(yolo_line + '\n')
                    
                    class_names = [det['detection_data']['class_name'] for det in frame_detections]
                    print(f"Saved dataset frame: {image_filename} with {len(frame_detections)} annotations: {', '.join(set(class_names))}")
                else:
                    print(f"Failed to save dataset frame: {image_path}")
                    
        except Exception as e:
            print(f"Error in _save_dataset_frame_detections: {e}")
            import traceback
            traceback.print_exc()

    def close(self):
        """Close all open file handles"""
        try:
            # Stop session rotation timer
            self.session_rotation_stop_event.set()
            if hasattr(self, 'session_rotation_thread'):
                self.session_rotation_thread.join(timeout=2)
            
            # Stop system metrics
            if self.enable_system_metrics:
                self.system_metrics_stop_event.set()
                if self.system_metrics_thread:
                    self.system_metrics_thread.join(timeout=2)
                if self.system_metrics_csv:
                    self.system_metrics_csv["file"].close()
                    
            # Close CSV files
            for file_info in self.csv_files.values():
                file_info["file"].close()
            self.csv_files.clear()
            
            print("InferenceSaver closed all files and stopped all threads")
            
        except Exception as e:
            print(f"Error during InferenceSaver cleanup: {e}")
            import gc
            gc.collect()
            print("Memory garbage collection completed")
    
    def register_camera(self, index, source_uri):
        """
        Register a camera with the inference saver.
        """
        # Map the source URI to a camera name
        camera_name = self.map_source_to_camera_name(index, source_uri)
        
        # Set up the stream (this will create directories and CSV files)
        self.setup_stream(index, source_uri)
        
        print(f"Registered camera {index} with URI {source_uri} as '{camera_name}'")
        return camera_name
