#!/bin/bash

# Define source and destination directories
SOURCE_DIR="motion_frame_high_quality"
DEST_DIR="game_animation"

# Create the root of the destination directory
mkdir -p "$DEST_DIR/characters"

# Iterate over each character in the source directory
for character in "$SOURCE_DIR"/*; do
    if [ -d "$character" ]; then
        character_name=$(basename "$character")
        echo "Processing character: $character_name"

        # Create character directory in destination
        mkdir -p "$DEST_DIR/characters/$character_name/motions"

        # Move and rename main.png to main_reference.png
        if [ -f "$character/main.png" ]; then
            cp "$character/main.png" "$DEST_DIR/characters/$character_name/main_reference.png"
        else
            echo "Warning: main.png not found for character $character_name"
        fi

        # Iterate over each motion in the character directory
        for motion in "$character"/*; do
            if [ -d "$motion" ]; then
                motion_name=$(basename "$motion")
                echo "  Processing motion: $motion_name"

                # Create motion directory in destination
                mkdir -p "$DEST_DIR/characters/$character_name/motions/$motion_name/poses"
                mkdir -p "$DEST_DIR/characters/$character_name/motions/$motion_name/ground_truth"

                # Move pose images
                mv "$motion"/humanpose_*.png "$DEST_DIR/characters/$character_name/motions/$motion_name/poses/" 2>/dev/null

                # Move ground truth frames
                mv "$motion"/frame_*.png "$DEST_DIR/characters/$character_name/motions/$motion_name/ground_truth/" 2>/dev/null

                # Copy the reference image (use main_reference.png or a specific reference)
                cp "$DEST_DIR/characters/$character_name/main_reference.png" "$DEST_DIR/characters/$character_name/motions/$motion_name/reference.png"
            fi
        done
    fi
done

echo "Rearrangement complete."
