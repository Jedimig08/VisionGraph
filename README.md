# VisionGraph
A low-cost camera-feedback polargraph controlled by a phone and ESP32.

## Concept

VisionGraph uses a camera to continuously determine the position
of the pen. The phone performs the computer vision, polargraph
kinematics, and motor calculations.

The ESP32 acts as a simple motor controller.

## Architecture

Phone
├── Camera
├── Computer Vision
├── Perspective Correction
├── Pen Tracking
├── Polargraph Mathematics
└── Motor Commands
        │
        ↓
      ESP32
        │
        ↓
    H-Bridge
      /   \
 Motor L  Motor R

## Status

🚧 Early development