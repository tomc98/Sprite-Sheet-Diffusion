import cv2
import os
import json
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

# Input and output image paths
image_path = "cute_dino/Walk/Walk (8).png"
output_resize_path = "cute_dino/Walk/frame_7.png"
output_path = "cute_dino/Walk/humanpose_7.png"

# Keypoints for the OpenPose COCO model and their respective colors (in BGR format)
keypoints = ["nose", "neck", "right_shoulder", "right_elbow", "right_wrist", 
             "left_shoulder", "left_elbow", "left_wrist", "right_hip", 
             "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
             "right_eye", "left_eye", "right_ear", "left_ear"]

# Colors for each keypoint (Hex converted to BGR)
keypoint_colors = [
    (0, 0, 255), (1, 85, 255), (1, 170, 255), (0, 255, 255),
    (3, 255, 170), (0, 255, 86), (0, 255, 3), (85, 255, 3),
    (171, 255, 3), (255, 255, 3), (255, 170, 5), (255, 85, 0),
    (255, 0, 0), (255, 0, 84), (255, 0, 170), (255, 0, 255),
    (169, 0, 255), (85, 0, 255)
]

# Skeleton connections and their respective colors (in BGR format)
skeleton_with_colors = [
    ((0, 1), (154, 0, 0)),      # neck to nose
    ((1, 2), (0, 0, 153)),      # neck to right shoulder
    ((1, 5), (1, 51, 153)),     # neck to left shoulder
    ((1, 8), (153, 153, 0)),    # neck to right hip
    ((1, 11), (153, 153, 0)),   # neck to left hip
    ((2, 3), (0, 101, 153)),    # right shoulder to right elbow
    ((3, 4), (0, 153, 153)),    # right elbow to right wrist
    ((5, 6), (0, 153, 101)),    # left shoulder to left elbow
    ((6, 7), (0, 153, 51)),     # left elbow to left wrist
    ((8, 9), (51, 153, 0)),     # right hip to right knee
    ((9, 10), (102, 153, 0)),   # right knee to right ankle
    ((11, 12), (153, 102, 0)),  # left hip to left knee
    ((12, 13), (153, 51, 0)),   # left knee to left ankle
    ((0, 14), (153, 0, 51)),    # nose to right eye
    ((0, 15), (153, 0, 153)),   # nose to left eye
    ((14, 16), (102, 0, 153)),  # right eye to right ear
    ((15, 17), (102, 0, 153))   # left eye to left ear
]

# Define diameter for keypoints and thickness for limbs
point_diameter = 8
limb_thickness = point_diameter

# Initialize the Tkinter window
root = tk.Tk()
root.title("Pose Annotation Tool")

# Load the image for annotation display
img = cv2.imread(image_path)
img = cv2.resize(img, (512, 512))
cv2.imwrite(image_path, img)
os.rename(image_path, output_resize_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_pil = Image.fromarray(img_rgb)
img_tk = ImageTk.PhotoImage(img_pil)

# Create Tkinter canvas and display the original image
canvas = tk.Canvas(root, width=img_pil.width, height=img_pil.height)
canvas.pack()
canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

# Label on the canvas to show the current keypoint name
instruction_label = tk.Label(root, text=f"Click to annotate: {keypoints[0]} or press S to skip", font=("Arial", 16))
instruction_label.pack()

# Initialize annotation data and skipped keypoints
annotations = {}
skipped_keypoints = set()
current_keypoint_index = 0

# Define the callback function for mouse click event
def on_click(event):
    global current_keypoint_index
    if current_keypoint_index < len(keypoints):
        keypoint_name = keypoints[current_keypoint_index]
        annotations[keypoint_name] = (event.x, event.y)
        
        # Display the clicked point
        canvas.create_oval(event.x - 3, event.y - 3, event.x + 3, event.y + 3, fill="red", outline="")
        canvas.create_text(event.x, event.y - 10, text=keypoint_name, fill="red")
        
        current_keypoint_index += 1
        # Update the instruction label to show the next keypoint name
        if current_keypoint_index < len(keypoints):
            instruction_label.config(text=f"Click to annotate: {keypoints[current_keypoint_index]} or press S to skip")
        else:
            instruction_label.config(text="All keypoints annotated! You may close the window and save the data.")

# Define the callback function to skip keypoint annotation
def skip_keypoint(event):
    global current_keypoint_index
    if current_keypoint_index < len(keypoints):
        keypoint_name = keypoints[current_keypoint_index]
        skipped_keypoints.add(keypoint_name)  # Mark the current keypoint as skipped
        
        current_keypoint_index += 1
        # Update the instruction label to show the next keypoint name
        if current_keypoint_index < len(keypoints):
            instruction_label.config(text=f"Click to annotate: {keypoints[current_keypoint_index]} or press S to skip")
        else:
            instruction_label.config(text="All keypoints annotated! You may close the window and save the data.")

# Bind the click event to the canvas
canvas.bind("<Button-1>", on_click)
# Bind the skip event to the 'S' key
root.bind("<Key-s>", skip_keypoint)

# Start the Tkinter main loop
root.mainloop()

# Create a black background image
black_background = np.zeros((img_pil.height, img_pil.width, 3), dtype=np.uint8)

# Draw limbs on the black background
for (start_end, color) in skeleton_with_colors:
    start, end = start_end
    # Check if keypoints have been annotated and not skipped
    if keypoints[start] in annotations and keypoints[end] in annotations:
        point_a = annotations[keypoints[start]]
        point_b = annotations[keypoints[end]]
        # Draw limb as an ellipse, with short axis equal to the diameter of keypoints
        center = ((point_a[0] + point_b[0]) // 2, (point_a[1] + point_b[1]) // 2)
        axis_length = (int(np.linalg.norm(np.array(point_a) - np.array(point_b)) / 2), limb_thickness // 2)
        angle = np.degrees(np.arctan2(point_b[1] - point_a[1], point_b[0] - point_a[0]))
        cv2.ellipse(black_background, center, axis_length, angle, 0, 360, color, thickness=-1)

# Draw keypoints
for idx, (keypoint_name, position) in enumerate(annotations.items()):
    color = keypoint_colors[idx]
    cv2.circle(black_background, position, point_diameter // 2, color, -1)

# Save the annotated image
cv2.imwrite(output_path, black_background)

# Save annotation data to JSON
# with open("annotations.json", "w") as f:
#     json.dump(annotations, f)

# print(f"Image saved to {output_path}, annotation data saved to annotations.json")
