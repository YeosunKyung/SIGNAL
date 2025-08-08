#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SIGNAL Model Demo
Quick demonstration of the SIGNAL model on a sample video

Author: Yeosun Kyung
Email: yeosun.kyung@yonsei.ac.kr
"""

import os
import sys
import numpy as np
from SIGNAL_model import SIGNALModel, SIGNALConfig

def demo_single_video(video_path, model_path=None):
    """
    Demo SIGNAL on a single video
    
    Args:
        video_path: Path to video file
        model_path: Path to saved model (optional)
    """
    print("="*60)
    print("🎬 SIGNAL Model Demo")
    print("="*60)
    
    # Initialize model
    model = SIGNALModel()
    
    if model_path and os.path.exists(model_path):
        # Load pre-trained model
        model.load_model(model_path)
        print(f"✅ Loaded model from {model_path}")
    else:
        print("⚠️  No pre-trained model found. Please train first.")
        return
    
    # Process single video
    print(f"\n📹 Processing video: {video_path}")
    features = model.processor.process_video(video_path)
    
    if features is None:
        print("❌ Failed to process video")
        return
    
    # Make prediction
    features = features.reshape(1, -1)  # Add batch dimension
    prediction = model.predict(features)
    
    # Action classes
    actions = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
    predicted_action = actions[prediction[0]]
    
    print(f"\n✨ Predicted action: {predicted_action}")
    print(f"📊 Feature vector size: {features.shape[1]} → 100 (after selection)")
    print(f"💾 Compression ratio: 14,400:1")
    
    return predicted_action

def demo_live_camera():
    """
    Demo SIGNAL on live camera feed
    Note: This is a simplified demo. Real-time processing would need optimization.
    """
    import cv2
    
    print("="*60)
    print("📷 SIGNAL Live Camera Demo")
    print("="*60)
    print("Press 'q' to quit, 'space' to capture action")
    
    # Initialize model (load pre-trained)
    model = SIGNALModel()
    if os.path.exists("signal_model.pkl"):
        model.load_model("signal_model.pkl")
    else:
        print("⚠️  No pre-trained model found. Please train first.")
        return
    
    # Open camera
    cap = cv2.VideoCapture(0)
    frames_buffer = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Show frame
        cv2.imshow('SIGNAL Demo - Press SPACE to analyze', frame)
        
        # Buffer frames
        frames_buffer.append(frame)
        if len(frames_buffer) > 100:  # Keep last 100 frames
            frames_buffer.pop(0)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' ') and len(frames_buffer) >= 30:
            print("\n🔍 Analyzing action...")
            
            # Process buffered frames
            features_list = []
            extractor = model.processor.extractor
            
            # Sample frame pairs
            indices = np.linspace(0, len(frames_buffer)-2, 30, dtype=int)
            for i in indices:
                features = extractor.extract_features(frames_buffer[i], frames_buffer[i+1])
                if features is not None:
                    features_list.append(features)
            
            if len(features_list) > 10:
                # Aggregate features
                features_array = np.array(features_list)
                aggregated = []
                aggregated.extend(np.mean(features_array, axis=0))
                aggregated.extend(np.std(features_array, axis=0))
                aggregated.extend(np.max(features_array, axis=0))
                
                # Predict
                features = np.array(aggregated).reshape(1, -1)
                prediction = model.predict(features[:, :model.processor.extractor.config.N_FEATURES_SELECTED])
                
                actions = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
                print(f"✨ Detected action: {actions[prediction[0]]}")
            
            # Clear buffer for next capture
            frames_buffer = frames_buffer[-30:]
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SIGNAL Model Demo')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--model', type=str, default='signal_model.pkl', help='Path to model file')
    parser.add_argument('--live', action='store_true', help='Use live camera')
    
    args = parser.parse_args()
    
    if args.live:
        demo_live_camera()
    elif args.video:
        demo_single_video(args.video, args.model)
    else:
        print("Usage:")
        print("  python demo.py --video path/to/video.avi --model signal_model.pkl")
        print("  python demo.py --live  # For live camera demo")