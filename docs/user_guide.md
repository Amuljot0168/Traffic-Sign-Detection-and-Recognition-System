# Traffic Sign Detection and Recognition System – User Guide

## Introduction

This guide provides step-by-step instructions to set up, build, and run the **Traffic Sign Detection and Recognition System**, a computer vision application built with **C++ and OpenCV**. It detects and classifies Indian traffic signs using **YOLOv5** and a **CNN-based classifier**.

---

## Installation

### Step 1: Clone the Repository

Open a terminal (PowerShell, Git Bash, or CMD) and run:

```bash
git clone https://github.com/Amuljot0168/Traffic-Sign-Detection-and-Recognition-System.git
cd Traffic-Sign-Detection-and-Recognition-System
```

---

### Step 2: Install Prerequisites

| Tool              | Description                                                                        |
| ----------------- | ---------------------------------------------------------------------------------- |
| MinGW-w64         | GCC/G++ compiler for Windows – install from [mingw-w64.org](https://mingw-w64.org) |
| CMake             | Cross-platform build tool – download from [cmake.org](https://cmake.org)           |
| Git               | Version control system – install from [git-scm.com](https://git-scm.com)           |
| Python (optional) | Python 3.x with pip – only needed for exporting YOLOv5 to ONNX                     |

After installing, make sure to **add MinGW and CMake’s `bin/` folders to your system PATH**.

---

## Configuration

### Step 3: OpenCV Setup

- Download the [OpenCV-MinGW-Build-OpenCV-4.5.5-x64](https://github.com/Amul24/OpenCV-download-File.git) version as a zip and extract it to your desired location.
- Add the OpenCV-MinGW-Build bin path to your System Environment Varible.
  
### Step 4: CMake Setup

- Open `setup_env_example.cmake` and update the OpenCV_DIR path.
- Rename the file to `setup_env.cmake`
- In your project root directory, run `cmake --build build` to make the build files

Update `setup_env.cmake`:

```cmake
set(OpenCV_DIR "C:/OpenCV-MinGW-Build-OpenCV-4.5.5-x64/x64/mingw/lib")
```

Change the path if your OpenCV folder is elsewhere.

---

### Step 5: Configure with CMake

```bash
cmake -G "MinGW Makefiles" -S . -B build/
```

This generates build files using the source in the current directory and stores them in the `build/` directory.

---

### Step 6: Build the Project

```bash
cmake --build build/ --config Release
```

This produces the `main.exe` executable in the `build/` directory.

---

## Running the Project

### Run with Image

```bash
cd build
main.exe --image path/to/image.jpg
```

### Run with Video

```bash
main.exe --video path/to/video.mp4
```

Replace the paths with your actual image/video files.


---

## Project Structure

```bash
Traffic-Sign-Detection-and-Recognition-System/
│
├── assets/demo/               # Demo images, videos, or media for documentation
│
├── config/                    # Configuration files (e.g., config.h)
│
├── data/                      # Input test data (e.g., images and videos)
│
├── docs/                      # Documentation files (e.g., tech_specs.md, user_guide.md)
│
├── include/TSDR/             # Header files for the project
│
├── models/                   # YOLOv5n ONNX model and class labels (e.g., coco.names)
│
├── python_scripts/           # Python utilities (e.g., for model export or preprocessing)
│
├── src/                      # Main C++ source code files (e.g., detection.cpp, utils.cpp)
│
├── tests/                   # Unit tests or test cases
│
├── .gitignore                  # Git ignore rules
├── CMakeLists.txt              # Main CMake build configuration
├── LICENSE                     # Project license
├── README.md                   # Project README file
├── main.cpp                    # Entry point of the program
├── setup_env_example.cmake     # CMake template for setting up environment paths

```

---

## Features

**Current:**

- Detects and classifies traffic signs in images/videos
- Uses YOLOv5 and CNN (exported to ONNX and XML)

**Coming Soon:**

- Live webcam support
- Alert system and GUI interface
- COMO Studio integration for mobile camera

---

## Troubleshooting

| Issue                            | Fix                                               |
| -------------------------------- | ------------------------------------------------- |
| OpenCV not found                 | Ensure correct path is set in `CMakeLists.txt`    |
| Model not loading                | Verify paths in `config/config.h` are correct     |
| Missing `.dll` files             | Copy OpenCV `.dll` files into the `build/` folder |
| Segfault on run                  | Make sure image/video input paths are valid       |
| CMake error: Generator not found | Ensure MinGW `bin/` is added to PATH              |

---
