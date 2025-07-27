import torch
import cv2
from pathlib import Path
from datetime import datetime
import os
import json
import time
from azure.iot.device import ProvisioningDeviceClient, IoTHubDeviceClient, Message
from datetime import datetime, timezone
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# === Load config from config.json ===
with open("config.json") as f:
    config = json.load(f)

RTSP_URL = config["rtsp_url"]
ID_SCOPE = config["azure"]["id_scope"]
DEVICE_ID = config["azure"]["device_id"]
DEVICE_KEY = config["azure"]["device_key"]

# === Connect to Azure IoT Central via DPS ===
provisioning_client = ProvisioningDeviceClient.create_from_symmetric_key(
    provisioning_host="global.azure-devices-provisioning.net",
    registration_id=DEVICE_ID,
    id_scope=ID_SCOPE,
    symmetric_key=DEVICE_KEY,
)
registration_result = provisioning_client.register()
assigned_hub = registration_result.registration_state.assigned_hub

device_client = IoTHubDeviceClient.create_from_symmetric_key(
    symmetric_key=DEVICE_KEY,
    hostname=assigned_hub,
    device_id=DEVICE_ID,
)
device_client.connect()
print(f"✅ Connected to Azure IoT Hub: {assigned_hub}")

# Force OpenCV to use TCP for RTSP with better settings
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;1024000|timeout;10000000|max_delay;500000"

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

# Target object classes
target_classes = {
    'person', 'bicycle', 'motorcycle', 'car', 'truck',
    'cat', 'sheep', 'cow'#, 'dog', 'elephant', 'horse', 'bear', 'zebra', 'giraffe'
}

# Output directory for video clips
output_dir = Path("detected_frames")
output_dir.mkdir(exist_ok=True)

def connect_rtsp(rtsp_url, max_attempts=3):
    """Try to connect to RTSP stream with multiple URL formats and backends"""
    
    # Try multiple URL formats for Tapo cameras
    url_variations = [
        rtsp_url,
        rtsp_url.replace(":554", ""),  # Remove port
        rtsp_url.replace("/stream1", "/stream2"),  # Try stream2
        "rtsp://192.168.0.109:554/stream1",  # Without credentials
        "rtsp://192.168.0.109/stream1",  # Without credentials and port
    ]
    
    print(f"🔗 Attempting to connect to RTSP...")
    
    for attempt in range(max_attempts):
        print(f"📡 Connection attempt {attempt + 1}/{max_attempts}")
        
        for url in url_variations:
            print(f"   Testing: {url}")
            
            # Try different backend options
            backends = [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
            
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(url, backend)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size
                    cap.set(cv2.CAP_PROP_FPS, 10)        # Set FPS
                    
                    if cap.isOpened():
                        # Test if we can actually read a frame
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            print(f"✅ Connected successfully!")
                            print(f"📺 Working URL: {url}")
                            return cap
                        else:
                            cap.release()
                except Exception as e:
                    continue
        
        if attempt < max_attempts - 1:
            print("⏳ Waiting 2 seconds before retry...")
            time.sleep(2)
    
    return None

# Open the RTSP stream with improved connection handling
cap = connect_rtsp(RTSP_URL)
if cap is None:
    print("❌ Could not open RTSP stream after multiple attempts.")
    print("🔧 Troubleshooting tips:")
    print("   1. Check if camera is online: ping 192.168.0.109")
    print("   2. Test RTSP URL in VLC: rtsp://taposerambi:1sampai8@192.168.0.109:554/stream1")
    print("   3. Verify camera RTSP is enabled in camera web interface")
    print("   4. Check username/password are correct")
    print("   5. Try alternative URLs: /stream2, without port :554")
    try:
        device_client.disconnect()
    except:
        pass
    exit()

print("📷 RTSP stream opened. Recording on detection...")

# Video writer init
writer = None
is_recording = False
no_object_frames = 0
max_no_object_frames = 30
detection_buffer_time = 2  # seconds to wait before starting new recording
last_detection_time = 0
first_frame_data = None  # Store first frame detection data for Azure telemetry

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 10
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

current_labels = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to read frame")
        continue

    results = model(frame)
    detections = results.xyxy[0]

    # Draw bounding boxes on frame
    annotated_frame = frame.copy()
    found_labels = set()
    
    for *box, conf, cls in detections.tolist():
        label = model.names[int(cls)]
        if label in target_classes:
            found_labels.add(label)
            
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box)
            confidence = conf
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label with confidence
            label_text = f"{label}: {confidence:.2f}"
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background rectangle for text
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            
            # Text
            cv2.putText(annotated_frame, label_text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    if found_labels:
        current_time = time.time()
        
        if not is_recording:
            # Check if enough time has passed since last detection ended
            time_since_last = current_time - last_detection_time
            
            if time_since_last >= detection_buffer_time:
                # Start new video and capture first frame data for Azure telemetry
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                label_str = "_".join(sorted(found_labels))
                filename = f"detection_{timestamp}_{label_str}.mp4"
                video_path = output_dir / filename

                # Print detected objects only when starting recording
                print(f"🧠 Detected: {', '.join(sorted(found_labels))}")
                print("🎥 Recording object")

                # Store first frame detection data for later Azure transmission
                first_frame_data = {
                    "timestamp": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
                    "objects": sorted(list(found_labels)),
                    "video_filename": filename,
                    "recording_start": timestamp
                }

                writer = cv2.VideoWriter(str(video_path), fourcc, fps, (frame_width, frame_height))
                is_recording = True
                current_labels = found_labels.copy()
            else:
                # Still in buffer period, skip recording
                remaining_buffer = detection_buffer_time - time_since_last
                print(f"⏳ Buffer period: {remaining_buffer:.1f}s remaining")
        else:
            # Add new labels to filename (in case they appear mid-recording)
            current_labels.update(found_labels)

        # Only write frame if we're actually recording
        if is_recording:
            writer.write(annotated_frame)  # Write frame with bounding boxes
        no_object_frames = 0

    else:
        if is_recording:
            no_object_frames += 1
            if no_object_frames > max_no_object_frames:
                # Finalize filename with updated labels
                writer.release()
                final_label_str = "_".join(sorted(current_labels))
                final_filename = f"detection_{timestamp}_{final_label_str}.mp4"
                final_path = output_dir / final_filename
                os.rename(video_path, final_path)
                print(f"⏹️ Recording stopped. Final file: {final_path}")
                
                # Update last detection time for buffer management
                last_detection_time = time.time()
                
                # Send telemetry to Azure ONLY after video is saved
                if first_frame_data:
                    try:
                        # Update payload with final information
                        first_frame_data["video_filename"] = final_filename
                        first_frame_data["recording_end"] = datetime.now().strftime("%Y%m%d_%H%M%S")
                        first_frame_data["final_objects_detected"] = sorted(list(current_labels))
                        first_frame_data["video_path"] = str(final_path)
                        
                        msg = Message(json.dumps(first_frame_data))
                        device_client.send_message(msg)
                        print(f"📡 Sent to Azure: {first_frame_data}")
                    except Exception as e:
                        print(f"❌ Failed to send Azure telemetry: {e}")
                    finally:
                        first_frame_data = None
                
                is_recording = False
                writer = None
                current_labels = set()

# Cleanup
if writer:
    writer.release()

# Send any pending telemetry before disconnecting
if first_frame_data:
    try:
        first_frame_data["video_filename"] = "incomplete_recording.mp4"
        first_frame_data["recording_status"] = "interrupted"
        msg = Message(json.dumps(first_frame_data))
        device_client.send_message(msg)
        print(f"📡 Final telemetry sent to Azure: {first_frame_data}")
    except Exception as e:
        print(f"❌ Failed to send final Azure telemetry: {e}")

try:
    device_client.disconnect()
    print("🔒 Disconnected from Azure IoT.")
except Exception as e:
    print(f"❌ Error disconnecting from Azure: {e}")

cap.release()
print("🔒 Stream closed.")
