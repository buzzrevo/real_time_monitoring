# from pypylon import pylon
# import cv2
# import numpy as np
# #import matplotlib.pyplot as plt
# import os
# import subprocess
# import time
# import requests
# import torch
# from torchvision import transforms
# import torchvision.models as models
# import torch.nn as nn
# import pickle
# from PIL import Image
# def shift_mask(mask, shift_pixels):
#     rows, cols = mask.shape
#     M_left = np.float32([[1, 0, -shift_pixels], [0, 1, 0]])
#     M_right = np.float32([[1, 0, shift_pixels], [0, 1, 0]])
#     shifted_left = cv2.warpAffine(mask, M_left, (cols, rows))
#     shifted_right = cv2.warpAffine(mask, M_right, (cols, rows))
#     return shifted_left, shifted_right
# def widen_mask(mask, shift_pixels):
#     shifted_left, shifted_right = shift_mask(mask, shift_pixels)
#     widened_mask = cv2.bitwise_or(mask, shifted_left)
#     widened_mask = cv2.bitwise_or(widened_mask, shifted_right)
#     return widened_mask
def model_evaluation(input_image,model):
    device = torch.device("cpu")
    input_image = input_image.to(device)
    # Perform inference
    with torch.no_grad():
        output = model(input_image)
    _, predicted_class = torch.max(output,1)
    print(f"Predicted class for {predicted_class.item()}")
    return predicted_class.item()
# def capture_and_analyze_video(avi_filename, mp4_filename, readout_filename, duration, frame_rate, time_it=False):
#     # Connect to the first available camera
#     camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
#     # Open the camera to access its node map
#     camera.Open()
#     # Set the pixel format to RGB8 if the camera supports it
#     if 'RGB8' in camera.PixelFormat.GetSymbolics():
#         camera.PixelFormat.SetValue('RGB8')
#     # Set the acquisition frame rate if supported
#     camera.AcquisitionFrameRate.SetValue(frame_rate)
#     camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
#     # Get the camera's resolution
#     width = camera.Width.GetValue()
#     height = camera.Height.GetValue()
#     # Set the region of interest
#     min_x = 50
#     max_x = 1080
#     min_y = 250
#     max_y = 810
#     width = max_x - min_x
#     height = max_y - min_y
#     # Define the codec and create a VideoWriter object
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(avi_filename, fourcc, frame_rate, (width, height))
#     # Load entire model
#     vgg16 = models.vgg16(pretrained=True)
#     num_classes = 3
#     vgg16.classifier[6] = nn.Linear(4096,num_classes)
#     checkpoint = torch.load('vgg16_printer_monitor_gcode_indicator_03_8.pth', map_location=torch.device('cpu'))
#     vgg16.load_state_dict(checkpoint['model_state_dict'])
#     vgg16.eval()
#     device = torch.device("cpu")
#     vgg16.to(device)
#     # Reapply transformation
#     transform = transforms.Compose([
#         transforms.Resize((224,224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     # Create container for extrusion rate queries along with a cotainer for logging times
#     filament_used_by_image = []
#     filament_used_by_image_timings = []
#     # Create class predicitions container along with container for logging times of image acquisition
#     prediciton_by_image = []
#     image_grabed_timings = []
#     # Create dictionary to handle readout
#     readout_info = {}
#     # Timing containers for timing analysis
#     if time_it:
#         requests_timing_container = []
#         analysis_timing_container = []
#         plotting_timing_container = []
#     # Retreive feed rate and update container
#     response = requests.get(f'http://192.168.1.20:7125/printer/objects/query?print_stats')
#     filament_used_by_image_timings.append(time.time())
#     print_stats = response.json()
#     filament_used = print_stats.get('result', {}).get('status', {}).get('print_stats', {}).get('filament_used', 'unknown')
#     filament_used_by_image.append(filament_used)
#     # Create a Matplotlib figure with subplots
#     #fig, axs = plt.subplots(1, 3, figsize=(15, 10))
#     #plt.ion()  # Enable interactive mode
#     # Capture frames and perform analysis
#     start_time = cv2.getTickCount()
#     grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)  # Initial frame
#     if grab_result.GrabSucceeded():
#         image_grabed_timings.append(time.time())
#         frame = grab_result.Array
#     # Flip u and down from camera orientation
#     frame = cv2.flip(frame, 0)
#     frame = frame[min_y:max_y,min_x:max_x,:]
#     while camera.IsGrabbing():
#         frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#         pil_image = Image.fromarray(frame)
#         input_image = transform(pil_image)
#         input_image = input_image.unsqueeze(0)
#         class_prediction = model_evaluation(input_image,vgg16)
#         prediciton_by_image.append(class_prediction)
#         grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
#         if grab_result.GrabSucceeded():
#             image_grabed_timings.append(time.time())
#             frame = grab_result.Array
#             frame = cv2.flip(frame, 0)
#             if time_it: timer_initial = time.time()
#             response = requests.get(f'http://192.168.1.20:7125/printer/objects/query?print_stats')
#             filament_used_by_image_timings.append(time.time())
#             print_stats = response.json()
#             filament_used = print_stats.get('result', {}).get('status', {}).get('print_stats', {}).get('filament_used', 'unknown')
#             filament_used_by_image.append(filament_used)
#             if time_it:
#                 requests_timing_container.append(time.time()-timer_initial)
#                 timer_initial = time.time()
#             frame = frame[min_y:max_y,min_x:max_x,:]
#             if time_it:
#                 analysis_timing_container.append(time.time()-timer_initial)
#                 timer_initial = time.time()
#             # Display using OpenCV
#             cv2.imshow("Original Frame", frame)
#             # Write the frame to the output file
#             out.write(frame)
#             if time_it: plotting_timing_container.append(time.time()-timer_initial)
#             # Break if the duration is exceeded
#             elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
#             if elapsed_time >= duration:
#                 break
#             # Break if 'q' is pressed
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         # Compile information into readout dict
#         readout_info['mp4_filename'] = mp4_filename
#         readout_info['filament_used_by_image'] = filament_used_by_image
#         readout_info['filament_used_by_image_timing'] = filament_used_by_image_timings
#         readout_info['prediciton_by_image'] = prediciton_by_image
#         readout_info['duration'] = duration
#         readout_info['frame_rate'] = frame_rate
#         # Dump with pickle to save
#         with open(readout_filename+'.pkl','wb') as f:
#             pickle.dump(readout_info, f)
#         if time_it: np.savetxt('timing_results.txt',np.vstack((requests_timing_container,analysis_timing_container,plotting_timing_container)))
#         grab_result.Release()
#     # Release resources
#     camera.StopGrabbing()
#     camera.Close()
#     out.release()
#     cv2.destroyAllWindows()
#     # Convert AVI to MP4 using FFmpeg
#     ffmpeg_command = [
#         'ffmpeg',
#         '-i', avi_filename,
#         '-c:v', 'libx264',
#         '-preset', 'ultrafast',
#         '-pix_fmt', 'yuv420p',
#         mp4_filename
#     ]
#     subprocess.run(ffmpeg_command)
# if __name__ == "__main__":
#     timestr = time.strftime("%Y%m%d-%H%M%S")
#     output_directory = 'C:\\Users\\revon\\OneDrive - Revo Foods\\tmp\\data\\raw'
#     avi_filename = f"{output_directory}\\output{timestr}.avi"
#     mp4_filename = f"{output_directory}\\output{timestr}.mp4"
#     readout_filename = f"{output_directory}\\output{timestr}"
#     duration = 160  # Duration in seconds
#     frame_rate = 10  # Set desired frame rate
#     capture_and_analyze_video(avi_filename, mp4_filename, readout_filename, duration, frame_rate, False)


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
    "method": "server.history.list",
    "params": {},
    "id": 5656
}

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
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
    duration = 160  # Duration in seconds
    frame_rate = 10  # Set desired frame rate
    capture_and_analyze_video(avi_filename, mp4_filename, readout_filename, duration, frame_rate, False)