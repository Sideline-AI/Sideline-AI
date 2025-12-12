# AI Sports Cam

An AI-powered sports camera system that automatically tracks and frames sports action using advanced computer vision and physics-based camera control. The project aims to create a mobile application that fundamentally improves the experience of watching amateur sports matches by automatically capturing professional-looking footage.

## 🎯 Project Overview

AI Sports Cam uses artificial intelligence to automatically track the ball and players on a sports field, intelligently frame the action, and record smooth, professional-quality videos. This eliminates the need for expensive equipment or manual camera operation.

**Current Status**: Phase 1 - Python Prototype ✅  
**Future Goal**: Flutter mobile app for Android and iOS

## ✨ Key Features

### Current Implementation (Prototype)
- **Dual-Model AI Detection**
  - Player detection using YOLOv8n (excellent for people)
  - Specialized ball detection using custom Roboflow model (~90% accuracy)
  - Intelligent merging of detections from both models

- **Physics-Based Camera Control (BroadcastCam)**
  - Mass-Spring-Damper physics for smooth, realistic camera movement
  - Active cluster tracking (60% ball + 40% nearest 3 players)
  - Y-axis lock for stable horizontal framing
  - Conservative zoom strategy optimized for 5v5 football

- **Perspective Correction**
  - Interactive perspective calibration for corner-mounted cameras
  - Real-world coordinate mapping using homography transformation

- **Real-Time Processing**
  - Frame skipping optimization (AI inference every 3 frames)
  - GPU acceleration support (CUDA)
  - Configurable performance/quality trade-offs

- **Debug & Development Tools**
  - Split-screen debug view (raw frame + broadcast output)
  - Real-time physics metrics display
  - Detection visualization

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenCV
- Ultralytics YOLOv8
- NumPy

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd jpb
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the custom ball detection model** (one-time setup)
   ```bash
   python setup_ball_model.py
   ```

### Usage

#### Basic Video Processing
```bash
python prototype.py --source video.mp4 --debug --min_zoom 1.2
```

#### Webcam (Real-time)
```bash
python prototype.py --source 0 --debug --min_zoom 1.2 --device cuda
```

#### Save Output Video
```bash
python prototype.py --source video.mp4 --save --debug
```

#### GPU Acceleration
```bash
python prototype.py --source video.mp4 --device cuda --debug
```

#### Performance Optimization
```bash
# High accuracy (slower)
python prototype.py --source video.mp4 --model yolov8s.pt --imgsz 640 --debug

# Fast performance (CPU)
python prototype.py --source video.mp4 --imgsz 480 --conf 0.2 --debug
```

For more detailed usage instructions, see [QUICK_START.md](QUICK_START.md).

## 📁 Project Structure

```
jpb/
├── prototype.py              # Main application ⭐
├── main.py                   # Alternative implementation
├── setup_ball_model.py       # Download custom ball model
├── requirements.txt          # Python dependencies
│
├── Models/
│   ├── yolov8n.pt           # Player detection model
│   ├── yolov8s.pt           # Alternative player model
│   └── roboflow_model/      # Custom ball detection model (after setup)
│       └── weights/
│           └── best.pt
│
├── Documentation/
│   ├── instructions.md                  # Project specifications (Polish)
│   ├── QUICK_START.md                  # Quick reference guide
│   ├── SYSTEM_OVERVIEW.md              # Complete system overview
│   ├── BroadcastCam_Documentation.md   # Camera physics explained
│   ├── Dual_Model_System.md            # Dual-model inference guide
│   └── Split_Screen_Debug_View.md      # Debug view documentation
│
└── Videos/                   # Test videos and outputs
    └── match_recording_*.mp4
```

## 🛠️ Technology Stack

### Current Prototype (Phase 1)
- **AI/ML**: Python, Ultralytics YOLOv8, TensorFlow Lite
- **Computer Vision**: OpenCV
- **Physics Simulation**: Custom Mass-Spring-Damper system
- **Video Processing**: OpenCV VideoWriter

### Planned Mobile App (Phase 2)
- **Frontend**: Flutter, Dart
- **Native Modules**: Kotlin (Android), Swift (iOS)
- **AI Runtime**: TensorFlow Lite

### Planned Backend (Phase 3)
- **API**: Python, FastAPI
- **Database**: PostgreSQL
- **Cloud Storage**: Google Cloud Platform (Cloud Storage)
- **DevOps**: Docker, Git

## 🎮 Key Components

### 1. Dual-Model Inference System
- Uses separate specialized models for players and ball detection
- Automatically falls back to single-model mode if ball model unavailable
- Provides ~90% ball detection accuracy (vs ~50% with single model)

### 2. BroadcastCam - Virtual Camera System
- Physics-based camera movement for natural, smooth framing
- Intelligent target selection balancing ball and player positions
- Configurable zoom limits and movement constraints

### 3. Perspective Transformer
- Corrects for camera angle and field perspective
- Maps pixel coordinates to real-world coordinates
- Interactive calibration tool

## 📊 Performance

| Feature | Single-Model | Dual-Model |
|---------|-------------|------------|
| Ball Detection | ~50% | ~90% |
| Player Detection | ✅ Excellent | ✅ Excellent |
| Speed (CPU) | 30 FPS | 20-25 FPS |
| Speed (GPU) | 60+ FPS | 50+ FPS |

## 🔄 Development Roadmap

### ✅ Phase 1: Python Prototype (Current)
- [x] AI detection system (dual-model)
- [x] Physics-based camera control
- [x] Video processing pipeline
- [x] Perspective correction
- [x] Debug tools

### 📱 Phase 2: Mobile Application (In Progress)
- [ ] Flutter UI/UX implementation
- [ ] Camera access and preview
- [ ] Native AI module integration (Kotlin/Swift)
- [ ] TensorFlow Lite optimization
- [ ] Real-time processing on mobile device

### ☁️ Phase 3: Backend & Cloud (Planned)
- [ ] User authentication system
- [ ] Cloud storage integration
- [ ] Video archiving and sharing
- [ ] User account management

## 🐛 Troubleshooting

**Problem**: "Custom ball model not found"
```bash
# Solution: Run setup
python setup_ball_model.py
```

**Problem**: Slow performance
```bash
# Solution: Use GPU or reduce image size
python prototype.py --source video.mp4 --device cuda --imgsz 480
```

**Problem**: Poor ball detection
```bash
# Solution: Lower confidence threshold
python prototype.py --source video.mp4 --conf 0.1 --debug
```

## 📖 Documentation

- [QUICK_START.md](QUICK_START.md) - Quick reference guide
- [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md) - Complete system overview
- [BroadcastCam_Documentation.md](BroadcastCam_Documentation.md) - Camera physics system
- [Dual_Model_System.md](Dual_Model_System.md) - Dual-model inference documentation
- [instructions.md](instructions.md) - Project specifications (Polish)

## 📝 License

[Add your license information here]

## 🤝 Contributing

[Add contribution guidelines if applicable]

---

**Note**: This is an active development project. The prototype is functional and being refined, with mobile app development planned for the next phase.

