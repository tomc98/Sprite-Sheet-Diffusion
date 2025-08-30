#!/usr/bin/env python3
"""Test animation preview and GIF export features"""

import requests
import time
import json
from PIL import Image
import io
import os

# Create a test image if it doesn't exist
test_image_path = '/tmp/test_character.png'
if not os.path.exists(test_image_path):
    img = Image.new('RGB', (256, 256), color='lightblue')
    img.save(test_image_path)
    print(f"Created test image: {test_image_path}")

print("Testing Animation Preview and GIF Export")
print("=" * 40)

# Upload with animation type
animation_type = 'walk'  # 8 frames
print(f"Testing {animation_type} animation (8 frames)...")

with open(test_image_path, 'rb') as f:
    files = {'file': ('test_character.png', f, 'image/png')}
    data = {'animation_type': animation_type}
    response = requests.post("http://localhost:8080/api/upload", files=files, data=data)

if response.status_code == 200:
    result = response.json()
    job_id = result['job_id']
    print(f"Job ID: {job_id}")
    
    # Poll for completion
    for i in range(10):
        time.sleep(1)
        status_response = requests.get(f"http://localhost:8080/api/status/{job_id}")
        if status_response.status_code == 200:
            status = status_response.json()
            if status['status'] == 'completed':
                print(f"✅ Animation completed!")
                print(f"Output URL: {status['output_url']}")
                
                # Test GIF export
                print("\nTesting GIF export...")
                gif_data = {
                    'job_id': job_id,
                    'frame_delay': '50'
                }
                gif_response = requests.post("http://localhost:8080/api/export-gif", data=gif_data)
                
                if gif_response.status_code == 200:
                    # Save the GIF
                    gif_path = f'/tmp/animation_{job_id}.gif'
                    with open(gif_path, 'wb') as gif_file:
                        gif_file.write(gif_response.content)
                    
                    # Verify it's a valid GIF
                    gif_img = Image.open(gif_path)
                    print(f"✅ GIF exported successfully!")
                    print(f"GIF saved to: {gif_path}")
                    print(f"GIF format: {gif_img.format}")
                    print(f"GIF size: {gif_img.size}")
                    
                    # Check if it's animated
                    try:
                        gif_img.seek(1)
                        print(f"✅ GIF is animated (has multiple frames)")
                    except EOFError:
                        print(f"⚠️ GIF appears to have only one frame")
                else:
                    print(f"❌ GIF export failed: {gif_response.status_code}")
                    if gif_response.headers.get('content-type') == 'application/json':
                        print(f"Error: {gif_response.json()}")
                
                break
            elif status['status'] == 'error':
                print(f"❌ Animation failed: {status['message']}")
                break
else:
    print(f"❌ Upload failed: {response.status_code}")
    if response.headers.get('content-type') == 'application/json':
        print(f"Error: {response.json()}")

print("\nTest complete!")
print("\nTo view the animation preview:")
print("1. Open http://localhost:8080 in your browser")
print("2. Upload an image and select an animation type")
print("3. After generation, you'll see:")
print("   - The sprite sheet")
print("   - An animated preview that plays the frames")
print("   - Speed control slider (10-200ms per frame)")
print("   - Play/Pause button")
print("   - Download as GIF button")