import time
import cv2
import pickle
import numpy as np
import subprocess
import websocket
import json
import threading
from pypylon import pylon
from PIL import Image
import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn

# Global variables to store the latest print stats
latest_print_stats = {}
lock = threading.Lock()

request = {
    "jsonrpc": "2.0",
    "method": "printer.objects.query",
    "params": {
        "objects": {
            "webhooks": null,
            "virtual_sdcard": null,
            "print_stats": null
        }
    },
    "id": 5664
}

def model_evaluation(input_image,model):
    device = torch.device("cpu")
    input_image = input_image.to(device)
    # Perform inference
    with torch.no_grad():
        output = model(input_image)
    _, predicted_class = torch.max(output,1)
    print(f"Predicted class for {predicted_class.item()}")
    return predicted_class.item()

def on_message(ws, message):
    global latest_print_stats
    #print(f"Received message: {message}")  # Debugging statement
    data = json.loads(message)
    if 'result' in data.keys():
        data = data['result']['jobs'][0] #json.loads(message)
        #print("before lock: ",data)
        with lock:
            #print("request with id 5656 received")
            latest_print_stats = data
            #print(f"Updated latest_print_stats: {latest_print_stats}")  # Debugging statement
        # Send the request again to keep receiving updates
    ws.send(json.dumps(request))

def on_error(ws, error):
    print(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("WebSocket connection closed")

def on_open(ws):
    print("WebSocket connection opened")
    ws.send(json.dumps(request))

# Start the WebSocket connection in a separate thread
ws = websocket.WebSocketApp("ws://192.168.1.20:7125/websocket",
                            on_open=on_open,
                            on_message=on_message,
                            on_error=on_error,
                            on_close=on_close)

ws_thread = threading.Thread(target=ws.run_forever)
ws_thread.start()

def capture_and_analyze_video(avi_filename, mp4_filename, readout_filename, duration, frame_rate, time_it):
    # Initialize camera
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    camera.PixelFormat.SetValue('RGB8') 

    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    # Define video codec and create VideoWriter object
    #frame_width = int(camera.Width.GetValue())
    #frame_height = int(camera.Height.GetValue())
    min_x = 50
    max_x = 1080
    min_y = 250
    max_y = 810
    frame_width = max_x - min_x
    frame_height = max_y - min_y
    out = cv2.VideoWriter(avi_filename, cv2.VideoWriter_fourcc('M','J','P','G'), frame_rate, (frame_width, frame_height))

    # Initialize variables
    plotting_timing_container = []
    filament_used_by_image_timings = []
    filament_used_by_image = []
    prediciton_by_image = []
    requests_timing_container = []
    analysis_timing_container = []
    start_time = cv2.getTickCount()

    # Wait for the first WebSocket message to be received
    while not latest_print_stats:
        print("Waiting for WebSocket message...")  # Debugging statement
        time.sleep(1)  # Add a small delay to avoid busy-waiting

    # Load entire model
    vgg16 = models.vgg16(pretrained=True)
    num_classes = 3
    vgg16.classifier[6] = nn.Linear(4096,num_classes)
    checkpoint = torch.load('vgg16_printer_monitor_gcode_indicator_03_8.pth', map_location=torch.device('cpu'))
    vgg16.load_state_dict(checkpoint['model_state_dict'])
    vgg16.eval()
    device = torch.device("cpu")
    vgg16.to(device)
    # Reapply transformation
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print('starting main loop')
    # Main loop
    while camera.IsGrabbing():
        timer_initial = time.time()
        grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grab_result.GrabSucceeded():
            frame = grab_result.Array
            frame = cv2.flip(frame, 0)
            pil_image = Image.fromarray(frame)
            input_image = transform(pil_image)
            input_image = input_image.unsqueeze(0)
            # Model evaluation (assuming model_evaluation and vgg16 are defined elsewhere)
            class_prediction = model_evaluation(input_image, vgg16)
            prediciton_by_image.append(class_prediction)
            
            with lock:
                filament_used = latest_print_stats['filament_used']
                #filament_used = latest_print_stats.get('filament_used', 'unknown')
            filament_used_by_image_timings.append(time.time())
            filament_used_by_image.append(filament_used)
            
            # Display using OpenCV
            cv2.imshow("Original Frame", frame)
            # Write the frame to the output file
            out.write(frame)
            if time_it: plotting_timing_container.append(time.time() - timer_initial)
            
            # Break if the duration is exceeded
            elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            if elapsed_time >= duration:
                break
            # Break if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Compile information into readout dict
    readout_info = {
        'mp4_filename': mp4_filename,
        'filament_used_by_image': filament_used_by_image,
        'filament_used_by_image_timing': filament_used_by_image_timings,
        'prediciton_by_image': prediciton_by_image,
        'duration': duration,
        'frame_rate': frame_rate
    }

    # Dump with pickle to save
    with open(readout_filename + '.pkl', 'wb') as f:
        pickle.dump(readout_info, f)

    if time_it:
        np.savetxt('timing_results.txt', np.vstack((requests_timing_container, analysis_timing_container, plotting_timing_container)))

    # Release resources
    grab_result.Release()
    camera.StopGrabbing()
    camera.Close()
    out.release()
    cv2.destroyAllWindows()

    # Convert AVI to MP4 using FFmpeg
    ffmpeg_command = [
        'ffmpeg',
        '-i', avi_filename,
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-pix_fmt', 'yuv420p',
        mp4_filename
    ]
    subprocess.run(ffmpeg_command)

if __name__ == "__main__":
    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_directory = 'C:\\Users\\revon\\OneDrive - Revo Foods\\tmp\\data\\raw'
    avi_filename = f"{output_directory}\\output{timestr}.avi"
    mp4_filename = f"{output_directory}\\output{timestr}.mp4"
    readout_filename = f"{output_directory}\\output{timestr}"
    duration = 20 #160 # Duration in seconds
    frame_rate = 10  # Set desired frame rate
    capture_and_analyze_video(avi_filename, mp4_filename, readout_filename, duration, frame_rate, False)