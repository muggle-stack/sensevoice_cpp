#!/bin/bash

# ASR C++ Build Script
set -e

echo "Building ASR C++ Application"
echo "==============================="

# Function to check for a command's existence
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "Error: Required command '$1' not found."
        echo "Please install it before proceeding."
        exit 1
    fi
}

# Function to check for a pkg-config package's existence
check_pkg_config_package() {
    if ! pkg-config --exists "$1"; then
        echo "Error: Required development package '$1' not found."
        echo "Please install it before proceeding."
        echo "  - For macOS (Homebrew): brew install $2"
        echo "  - For Ubuntu/Debian: sudo apt install $3"
        exit 1
    fi
}

echo "Checking essential build dependencies..."

# --- Check for Compilers (GCC/G++) ---
# 在macOS上，通常使用Clang/LLVM，而不是gcc-14/g++-14。
# 如果项目强制要求GCC，则用户需要通过brew安装。
# 这里改为检查系统默认的g++和gcc，对于macOS，它们通常链接到Clang。
if command -v g++ &> /dev/null; then
    echo "G++ Compiler found."
else
    echo "Error: G++ Compiler not found."
    echo "Please install a C++ compiler (e.g., Xcode Command Line Tools on macOS)."
    echo "  - For macOS: xcode-select --install"
    exit 1
fi

if command -v gcc &> /dev/null; then
    echo "GCC Compiler found."
else
    echo "Error: GCC Compiler not found."
    echo "Please install a C compiler (e.g., Xcode Command Line Tools on macOS)."
    echo "  - For macOS: xcode-select --install"
    exit 1
fi


# --- Check for CMake ---
check_command "cmake"
CMAKE_VERSION=$(cmake --version | head -n1 | cut -d' ' -f3)
echo "CMake $CMAKE_VERSION found."
# Optional: enforce minimum CMake version
# if (( $(echo "$CMAKE_VERSION < 3.16" | bc -l) )); then
#     echo "Error: CMake version must be 3.16 or higher. Found $CMAKE_VERSION."
#     exit 1
# fi


# --- Check for pkg-config ---
# pkg-config is usually needed for C++ development on Linux/macOS to find libs
check_command "pkg-config"


# --- Check for PortAudio development files ---
# macOS: portaudio (Homebrew installs dev files)
# Ubuntu: libportaudio-dev
check_pkg_config_package "portaudio-2.0" "portaudio" "libportaudio-dev"
echo "PortAudio development files found."


# --- Check for libsndfile development files ---
# macOS: libsndfile (Homebrew installs dev files)
# Ubuntu: libsndfile1-dev
check_pkg_config_package "sndfile" "libsndfile" "libsndfile1-dev"
echo "libsndfile development files found."

# --- ADDED: Check for FFTW (fftw3f) development files ---
# FFTW is often used for Fast Fourier Transforms in audio processing
# macOS: fftw (Homebrew installs both float and double precision versions)
# Ubuntu: libfftw3-dev (installs both float and double precision)
# pkg-config check for fftw3f or fftw3
if ! pkg-config --exists "fftw3f" && ! pkg-config --exists "fftw3"; then
    echo "Error: FFTW (Fastest Fourier Transform in the West) development files not found."
    echo "Your project likely needs libfftw3f (single precision) or libfftw3 (double precision)."
    echo "Please install FFTW development files."
    echo "  - For macOS (Homebrew): brew install fftw"
    echo "  - For Ubuntu/Debian: sudo apt install libfftw3-dev"
    exit 1
fi
echo "FFTW development files found."


# --- Check for ONNX Runtime headers (specific file check as pkg-config might not cover it universally) ---
ONNXRUNTIME_FOUND=false
# Paths for macOS (Homebrew) and Linux common install locations
for path in /usr/local/include/onnxruntime /usr/include/onnxruntime /opt/homebrew/include/onnxruntime /usr/local/include /usr/include; do
    if [ -f "$path/onnxruntime_c_api.h" ]; then
        ONNXRUNTIME_FOUND=true
        echo "ONNX Runtime headers found at $path"
        break
    fi
done

if [ "$ONNXRUNTIME_FOUND" = false ]; then
    echo "Error: ONNX Runtime C++ library headers not found."
    echo "Please install ONNX Runtime C++ library."
    echo "  - For macOS: brew install onnxruntime"
    echo "  - For Ubuntu/Debian: Download from https://github.com/microsoft/onnxruntime/releases"
    echo "    and extract to a system path (e.g., /usr/local) or set ONNXRUNTIME_HOME environment variable."
    exit 1
fi


# --- Check for cURL development files ---
# macOS: curl (Homebrew installs dev files)
# Ubuntu: libcurl4-openssl-dev
check_pkg_config_package "libcurl" "curl" "libcurl4-openssl-dev"
echo "cURL development files found."

echo "All essential build dependencies checked."
echo ""

# Create build directory
echo "Creating build directory..."
mkdir -p build
cd build

# Configure with CMake
echo ""
echo "Configuring build..."
cmake ..

# Build
echo ""
echo "Building..."
# Use nproc for Linux, sysctl for macOS for parallel build jobs
MAKE_JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
echo "Using $MAKE_JOBS parallel jobs."
make -j$MAKE_JOBS

echo ""
echo "Build completed successfully!"
echo ""
echo "To run the application:"
echo "  cd build"
echo "  ./bin/asr_cpp"
echo ""
echo "Build artifacts:"
echo "  Executable: build/bin/asr_cpp"
echo "  Build logs: build/"
echo ""
echo "Next steps:"
echo "  1. Run './bin/asr_cpp' to start the demo"
echo "  2. Models will be downloaded automatically on first run"
echo "  3. Press Enter to start recording when prompted"
