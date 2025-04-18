import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_speed(flow, scale_factor, fps):
    """
    Estimate speed in km/h from optical flow vectors.
    """
    magnitudes = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)  # Compute magnitude of flow vectors
    avg_magnitude = np.mean(magnitudes)  # Take average movement
    speed_m_per_s = avg_magnitude * scale_factor * fps  # Convert pixels/frame to m/s
    speed_km_per_h = speed_m_per_s * 3.6  # Convert m/s to km/h
    return speed_km_per_h

def detect_vehicles(frame, fg_mask):
    """
    Detect vehicles using background subtraction and contour filtering.
    """
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    vehicle_contours = []
    
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Adjust threshold for vehicles
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            if 1.2 > aspect_ratio > 0.3:  # Filter out non-vehicle shapes
                vehicle_contours.append((x, y, w, h))
    
    return vehicle_contours

def classify_motion_layers(flow):
    """
    Classify motion into layers based on speed.
    """
    magnitudes = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    layers = np.digitize(magnitudes, bins=[2, 5, 10, 20])  # Define speed bins
    return layers

def draw_bounding_box(frame, vehicles, flow, scale_factor, fps, motion_layers):
    """
    Draw bounding boxes around detected vehicles and display their individual speeds.
    """
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Different colors for different layers
    
    for (x, y, w, h) in vehicles:
        roi_flow = flow[y:y+h, x:x+w]  # Extract flow for the detected vehicle
        roi_layer = motion_layers[y:y+h, x:x+w]  # Get motion layer
        
        speed = calculate_speed(roi_flow, scale_factor, fps)  # Compute individual vehicle speed
        layer_index = int(np.median(roi_layer))  # Determine layer
        color = colors[min(layer_index, len(colors) - 1)]
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{speed:.2f} km/h", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def plot_comparison(optical_flow_speeds, layered_motion_speeds):
    """
    Plot accuracy comparison of Optical Flow vs Layered Motion.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(optical_flow_speeds, label='Optical Flow', marker='o')
    plt.plot(layered_motion_speeds, label='Layered Motion', marker='s')
    plt.xlabel('Frame Index')
    plt.ylabel('Speed (km/h)')
    plt.title('Comparison of Optical Flow vs Layered Motion')
    plt.legend()
    plt.grid(True)
    plt.show()

def main(video_path, scale_factor, fps):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Unable to read video")
        return
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    background_subtractor = cv2.createBackgroundSubtractorMOG2()
    
    optical_flow_speeds = []
    layered_motion_speeds = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        motion_layers = classify_motion_layers(flow)
        
        # Apply background subtraction to detect moving objects
        fg_mask = background_subtractor.apply(gray)
        vehicles = detect_vehicles(frame, fg_mask)
        
        # Compute average speeds for comparison
        avg_optical_flow_speed = calculate_speed(flow, scale_factor, fps)
        avg_layered_motion_speed = np.mean([calculate_speed(flow[y:y+h, x:x+w], scale_factor, fps) for x, y, w, h in vehicles])
        
        optical_flow_speeds.append(avg_optical_flow_speed)
        layered_motion_speeds.append(avg_layered_motion_speed if not np.isnan(avg_layered_motion_speed) else 0)
        
        draw_bounding_box(frame, vehicles, flow, scale_factor, fps, motion_layers)
        
        # Display frame
        cv2.imshow('Layered Motion Estimation', frame)
        
        # Update previous frame
        prev_gray = gray.copy()
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    plot_comparison(optical_flow_speeds, layered_motion_speeds)

if __name__ == "__main__":
    video_path = r"C:\Users\hp\Downloads\r.mp4"  # Replace with your video file
    scale_factor = 0.05  # Approximate meters per pixel (adjust as needed)
    fps = 30  # Frames per second of the video
    main(video_path, scale_factor, fps)
