#!/usr/bin/env python3
"""Test different animation types"""

import requests
import time
import json

# Test animations
test_animations = ['idle', 'walk', 'jump', 'attack']

print("Testing Animation Selection Feature")
print("=" * 40)

# Check available animations
response = requests.get("http://localhost:8080/api/animations")
if response.status_code == 200:
    animations = response.json()
    print(f"Found {len(animations)} animation types:")
    for anim in animations:
        print(f"  - {anim['name']}: {anim['description']} ({anim['frames']} frames)")
    print()

# Test each animation type
for anim_type in test_animations:
    print(f"Testing {anim_type} animation...")
    
    # Upload with animation type
    with open('/tmp/test_character.png', 'rb') as f:
        files = {'file': ('test_character.png', f, 'image/png')}
        data = {'animation_type': anim_type}
        response = requests.post("http://localhost:8080/api/upload", files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        job_id = result['job_id']
        print(f"  Job ID: {job_id}")
        
        # Poll for completion
        for i in range(10):
            time.sleep(1)
            status_response = requests.get(f"http://localhost:8080/api/status/{job_id}")
            if status_response.status_code == 200:
                status = status_response.json()
                if status['status'] == 'completed':
                    print(f"  ✅ {anim_type} animation completed!")
                    print(f"  Output: {status['output_file']}")
                    break
                elif status['status'] == 'error':
                    print(f"  ❌ {anim_type} animation failed: {status['message']}")
                    break
    else:
        print(f"  ❌ Upload failed for {anim_type}")
    
    print()

print("Test complete!")