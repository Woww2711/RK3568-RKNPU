# YOLOv11 Face Detection on RK3568 (Adapted C++ Demo)

## 1. Project Goal

The primary objective is to run a custom-trained YOLOv11n face detection model (`yolov11n-face.pt`, trained on WIDER dataset) efficiently on an RK3568 development board, utilizing its NPU for acceleration. The desired output is a C++ application that takes an image path as input, performs inference using the NPU, and outputs an image with bounding boxes drawn around detected faces.

---

## 2. Prerequisites and Setup

This project involves development on a host PC and deployment to a target RK3568 board. Ensure the following requirements are met:

**2.1. Target Device (RK3568 Board)**

*   **Hardware:** A development board based on the Rockchip RK3568 SoC (e.g., Firefly Station M2, Radxa E25, etc.).
*   **Operating System:** A Linux distribution (like Ubuntu or Debian) provided by the board vendor, specifically built for the RK3568. Ensure the necessary kernel modules for the NPU (`rknpu`) and multimedia (RGA, VPU) are included and functional in the chosen OS image.
    *   *Note: As identified during troubleshooting, some vendor BSPs might have issues with NPU driver initialization. Using an image where the C++ SDK examples are known to work is recommended.*
*   **RKNN Runtime Libraries:** The target board needs the `librknnrt.so` runtime library installed in the appropriate system path (e.g., `/usr/lib/`). This is usually included in the vendor's OS image or can be installed separately from the RKNN SDK.
*   **ADB Access:** Android Debug Bridge (`adb`) is required for easy file transfer and remote shell access from the host PC. Ensure `adb` is enabled on the board and accessible from the host.

**2.2. Host Development PC**

*   **Operating System:** Linux (Ubuntu 18.04/20.04/22.04 recommended) or Windows 10/11 with WSL2 (Ubuntu distribution).
*   **Python Environment:**
    *   **Recommended:** Python 3.8.x using Miniforge or Anaconda/Miniconda for managing environments. While Python 3.10 was used in this project, compatibility issues can sometimes arise, and Rockchip documentation often defaults to 3.8.
        *   Miniforge Installer: [https://github.com/conda-forge/miniforge#download](https://github.com/conda-forge/miniforge#download)
    *   **Installation:** Create and activate a dedicated environment:
        ```bash
        conda create -n rknn_yolo python=3.8
        conda activate rknn_yolo
        ```
*   **RKNN Toolkit 2:**
    *   **Source:** https://github.com/airockchip/rknn-toolkit2
    *   **Version:** v2.3.2 was used during debugging. Newer versions might exist.
    *   **Download:** Typically obtained from Rockchip's official sources (Developer Wiki, SDK releases, or GitHub). Direct public links can change, refer to official Rockchip resources.
    *   **Installation (within activated Python env):**
        ```bash
        # Example using pip (adjust wheel file name/path as needed)
        pip install packages/rknn_toolkit2-*-cp38-cp38-linux_x86_64.whl

        # Or install from source if provided
        # pip install .
        ```
        *Refer to the `RKNN_SDK_Quick_Start_v2.3.2.pdf` (Section 3.2.2) for detailed installation options.*
*   **Model Conversion Libraries (Python):** Install within the activated Python environment:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu # Or GPU version if needed
    pip install ultralytics # For loading .pt and exporting ONNX
    pip install onnx onnxruntime # For ONNX handling
    pip install opencv-python numpy # General utilities
    ```
    *   *Note: Ensure PyTorch version compatibility with the `ultralytics` version used for training/exporting.*
*   **C++ Cross-Compilation Toolchain (for Linux Target):**
    *   **Requirement:** GCC toolchain for `aarch64-linux-gnu`.
    *   **Download:** Often provided within the RKNN SDK or downloadable from Linaro: [https://releases.linaro.org/components/toolchain/binaries/](https://releases.linaro.org/components/toolchain/binaries/) (e.g., `gcc-linaro-6.3.1-...-x86_64_aarch64-linux-gnu.tar.xz`).
    *   **Setup:** Extract the toolchain and ensure the `build-linux.sh` script points to the correct compiler path (e.g., `Projects/gcc-linaro-6.3.1.../bin/aarch64-linux-gnu-g++`).
*   **CMake:** Required for building the C++ application.
    ```bash
    sudo apt update
    sudo apt install cmake
    ```
*   **ADB Tools:**
    ```bash
    sudo apt update
    sudo apt install android-tools-adb android-tools-fastboot
    ```
*   **RKNN SDK:** (Needed for C++ development)
    *   **Download:** Typically obtained alongside RKNN Toolkit 2 from Rockchip's official sources.
    *   **Contents:** Contains the C++ API header (`rknn_api.h`), pre-compiled runtime libraries (`librknnrt.so` for different architectures), and potentially example code. The build scripts often rely on finding this SDK structure.
*   **`rknn_model_zoo` Repository:**
    *   **Source:** [https://github.com/airockchip/rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo)
    *   **Action:** Clone this repository, as the project structure and utility code (`utils/`, `3rdparty/`) are used.
        ```bash
        git clone https://github.com/airockchip/rknn_model_zoo.git
        cd rknn_model_zoo/examples/yolo11 # Navigate to the example directory
        ```

---

## 3. Background: Initial Approach & Problem

*   **Initial Attempt:** A Python script using the `rknn.api` library was developed to load a `.rknn` model (converted from the `.pt` file) and run inference directly on the RK3568 board.
*   **Core Problem:** The Python script failed during the NPU initialization step (`rknn.init_runtime()`). Error logs indicated the runtime environment couldn't properly detect or initialize the NPU hardware (`E init_runtime: RKNN model that loaded by 'load_rknn' not support inference on the simulator...`).
*   **Diagnosis:** Troubleshooting revealed issues with the NPU kernel driver (`rknpu`) loading correctly on multiple official vendor-provided Linux OS images (Ubuntu, Debian). Kernel messages (`dmesg`) pointed towards problems related to NPU power management definitions in the Device Tree Blob (DTB), specifically errors like `failed to find power_model node`. Despite DTB inspection showing relevant nodes present, the initialization failed, suggesting a subtle bug or configuration issue within the Board Support Package (BSP).
*   **SDK Success:** Crucially, C++ examples provided within Rockchip's official RKNN SDK (using `librknnrt.so`) *were* able to run successfully on the board, proving the NPU hardware and core libraries were functional.

## 4. Chosen Solution: Adapt C++ Example

Based on the successful execution of the SDK's C++ examples, the most reliable path forward was determined to be adapting an existing C++ example from the `rknn_model_zoo` repository for the face detection task.

The example `rknn_model_zoo/examples/yolo11` (originally for COCO object detection) was chosen as the base.

## 5. Model Conversion Process (Host PC - WSL2)

The custom `yolov11n-face.pt` model needed conversion to the `.rknn` format required by the NPU.

1.  **Intermediate Format:** Exported the PyTorch model (`.pt`) to ONNX (`.onnx`) using the `ultralytics` Python library.
    *   **Problem:** Initial ONNX export used dynamic input shapes (`['batch', 3, 'height', 'width']`).
    *   **Fix:** Re-exported the ONNX model with static input shapes (`[1, 3, 640, 640]`) by specifying `dynamic=False` during the export process. This generated `yolov11n-face.onnx`.
2.  **ONNX to RKNN Conversion:** Used the `convert.py` script (provided within the `yolo11/python` example directory, slightly modified) with the `rknn-toolkit2` library on the host PC (WSL2).
    *   **Target Platform:** `rk3568`
    *   **Quantization:** Enabled (`i8` - INT8 asymmetric).
    *   **Dataset:** A custom `face_dataset.txt` file listing paths (~50-100) to representative face images (full scenes, not just cropped faces) accessible from the WSL2 environment was used for quantization calibration.
    *   **Preprocessing:** Configured with `mean_values=[[0, 0, 0]]`, `std_values=[[255, 255, 255]]`.
    *   **Modifications to `convert.py`:**
        *   Updated `DATASET_PATH` to point to the custom `face_dataset.txt`.
        *   Changed `DEFAULT_RKNN_PATH` to `../model/yolov11n-face.rknn`.
        *   Changed `verbose=False` to `verbose=True` (for debugging conversion).
    *   **Input to Script:** The static-shape `yolov11n-face.onnx`.
    *   **Command Example:** `python convert.py ../model/yolov11n-face.onnx rk3568 i8 ../model/yolov11n-face.rknn`
    *   **Output:** `yolov11n-face.rknn` (placed in `yolo11/model/`).

## 6. C++ Code Adaptation (`rknn_model_zoo/examples/yolo11/cpp`)

The C++ code from the base `yolo11` example was modified significantly to work for single-class face detection and target the RK3568 platform cleanly.

**Modified Files:**

*   `model/coco_80_labels_list.txt`
*   `cpp/postprocess.h`
*   `cpp/yolo11.h`
*   `cpp/postprocess.cc`
*   `cpp/rknpu2/yolo11.cc`
*   `cpp/main.cc`
*   `cpp/CMakeLists.txt`

**Summary of Changes:**

1.  **Label File:** `model/coco_80_labels_list.txt` content replaced with a single line: `face`.
2.  **Single-Class Logic:**
    *   `postprocess.h`: Changed `OBJ_CLASS_NUM` define from 80 to 1.
    *   `postprocess.cc`: Simplified loops and logic related to class processing; removed multi-class NMS handling; updated `coco_cls_to_name` logic.
3.  **RK3568/RKNPU2 Focus (Code Cleanup):**
    *   `rknpu2/yolo11.cc`, `main.cc`, `yolo11.h`: Removed code blocks (`#if defined(...)`), variables (`rknn_dma_buf`, `rknn_tensor_mem`), and includes related to RV1106/1103 DMA, RKNPU1, and Zero Copy implementations.
    *   `CMakeLists.txt`: Renamed project; removed conditional compilation logic based on `TARGET_SOC` for other platforms; removed the separate `_zero_copy` build target; removed installation of `bus.jpg`.
4.  **Post-Processing Rewrite (Single Output Tensor):**
    *   `postprocess.cc`: The main `post_process` function was completely rewritten to handle the model's single output tensor (shape `[1, 5, 8400]`) instead of the original multi-tensor output structure. This involved directly decoding box coordinates (`cx, cy, w, h`) and confidence scores from the single buffer. The old `process_i8`, `process_fp32`, `compute_dfl` functions were removed/integrated. Added debug prints for confidence scores.
5.  **Bug Fixes During Adaptation:**
    *   **Type Definitions:** Fixed compilation errors related to conflicting declarations and incomplete types for `rknn_app_context_t` by ensuring correct include order and using `typedef struct rknn_app_context_t rknn_app_context_t;` forward declaration in `postprocess.h`.
    *   **Letterbox Coordinates:** Fixed errors in `postprocess.cc` where incorrect member names (`ori_width`/`ori_height`) were used for the `letterbox_t` struct. Modified `post_process` function signature and call site (`inference_yolo11_model` in `yolo11.cc`) to pass original image width/height explicitly.
    *   **RKNN Input Type:** Fixed runtime error `-5` from `rknn_inputs_set` by adding logic in `inference_yolo11_model` (`yolo11.cc`) to correctly detect when the model expects `INT8` input and explicitly converting the `UINT8` image data [0, 255] to the required `INT8` range [-128, 127] before passing the buffer to the RKNN API.
    *   **Padding Color:** Changed `bg_color` in `inference_yolo11_model` (`yolo11.cc`) from `114` to `0` (black).
    *   **Abort Crash:** Mitigated potential `malloc_consolidate` crash on exit by simplifying memory cleanup in `deinit_post_process`.
    *   **Threshold Change (Debug):** Lowered `BOX_THRESH` in `postprocess.h` to `0.05` temporarily to see low-confidence detections.

## 7. Compilation

*   **Environment:** Cross-compilation toolchain for `aarch64-linux-gnu` set up (as per RKNN SDK instructions).
*   **Command:** Build script from `rknn_model_zoo` root directory:
    ```bash
    ./build-linux.sh -t rk356x -a aarch64 -d yolo11
    ```
    *(Note: `-t rk356x` covers RK3566/RK3568. `-d yolo11` specifies the example directory name containing the modified `CMakeLists.txt`)*.
*   **Result:** Successful compilation after applying the fixes mentioned above.

## 8. Deployment & Execution

1.  **Locate:** Find the `install/rk356x_linux_aarch64/rknn_yolo11_face_demo/` directory.
2.  **Transfer:** Push the entire `rknn_yolo11_face_demo` directory to the RK3568 board (e.g., `/data/`).
3.  **Push Image:** Copy a test image (e.g., `test.jpg`) to the `model` subdirectory on the board.
4.  **Execute (on board via ADB shell):**
    ```bash
    cd /data/rknn_yolo11_face_demo/
    export LD_LIBRARY_PATH=./lib
    ./rknn_yolo11_face_demo model/yolov11n-face.rknn model/test.jpg
    ```

## 9. Current Status & Known Issues (As of Last Test)

*   **ONNX Host Test:** The static `yolov11n-face.onnx` model runs correctly using `onnxruntime` via the adapted `yolo11.py` script on the host PC (WSL2), successfully detecting faces. This validates the core model logic and the Python pre/post-processing.
*   **Python ADB FP32 Test:** The `yolov11n-face-fp32.rknn` model (converted using `rknn-toolkit2`) **runs successfully** via the Python script using `adb` and `rknn_server` on the RK3568, detecting faces correctly. This indicates the FP32 `.rknn` conversion is valid and runnable via the server interface.
*   **C++ FP32 Test:** The *same* `yolov11n-face-fp32.rknn` model **fails** when run using the compiled C++ application (linking `librknnrt.so` directly). It produces `0.0000` confidence scores for all potential detections, indicating an issue likely related to the C++ runtime environment or the direct `librknnrt.so` interaction (e.g., FP32/FP16 handling, format mismatch despite query, or library bug). The previous `malloc_consolidate` abort seems resolved.
*   **INT8 Tests (C++ and Python ADB):** All attempts to run INT8 `.rknn` models (quantized using default, MMSE, possibly per-channel methods) have **failed** on the device using both the C++ application and the Python ADB method. The raw INT8 confidence output from the NPU is consistently `-128` (the quantization zero-point), resulting in `0.0` dequantized confidence and no detections.

**Primary Conclusion:** The core issue preventing successful INT8 deployment appears to be a **failure in the INT8 quantization process** during the `.onnx` -> `.rknn` conversion using `rknn-toolkit2`. The generated INT8 `.rknn` models are producing invalid confidence outputs on the NPU. A secondary issue exists with running the valid FP32 `.rknn` model directly via C++/`librknnrt.so`.

## 10. Next Steps: Debugging Quantization & Conversion

The immediate focus must be on generating a **working INT8 `.rknn` model**. Debugging efforts should concentrate on the model conversion step on the host PC (WSL2).

1.  **Re-Examine Verbose INT8 Log:** Carefully analyze the full output log generated by `convert.py` with `verbose=True`. Look for warnings about quantization statistics (min/max ranges near output head), layers not quantized, data type defaults (e.g., FP16 warnings), saturation, or errors like "Unknown op target".
2.  **Simplify ONNX Model:** Use `onnx-simplifier` to potentially remove redundant nodes or structures that might interfere with quantization.
    ```bash
    # Install: pip install onnx-simplifier
    python -m onnxsim ../model/yolov11n-face.onnx ../model/yolov11n-face-simplified.onnx
    # Re-run INT8 conversion using the simplified ONNX file
    python convert.py ../model/yolov11n-face-simplified.onnx rk3568 i8 ../model/yolov11n-face-simplified-i8.rknn
    # Test the resulting -simplified-i8.rknn model on the device (C++ or Python ADB)
    ```
3.  **Verify/Confirm Preprocessing:** Double-check the exact preprocessing (normalization mean/std, channel order) expected by the original `.pt` model. Ensure the `mean_values` and `std_values` in `convert.py` precisely match this expectation.
4.  **Improve/Replace Quantization Dataset:** Ensure the `face_dataset.txt` contains 50-100+ diverse, high-quality, representative images with correct paths accessible from WSL2. Consider using a different dataset subset (e.g., from FDDB or WIDER) for calibration. Re-run conversion after updating the dataset.
5.  **Try Per-Channel Quantization:** If not already tested exhaustively, modify `convert.py` to use `quantized_method='channel'` in `rknn.config()` and re-run INT8 conversion.
6.  **Experiment with Older RKNN Toolkit:** If feasible, try converting the model using an older toolkit version (e.g., 1.6.0, 1.7.1) known to be stable for RK3568.
7.  **(Optional - Defer) Debug C++ FP32 Runtime:** Once a working INT8 model is achieved, the discrepancy between Python ADB FP32 and C++ FP32 execution can be revisited if necessary. This might involve testing NCHW input format in C++, ensuring correct FP16 handling if possible, or checking for `librknnrt.so` version issues.

## 11. Project File Overview (Key Files)

*   `model/yolov11n-face.rknn`: The quantized face detection model (Problematic INT8).
*   `model/yolov11n-face-fp32.rknn`: FP32 converted model (Works via Python ADB, fails via C++).
*   `model/yolov11n-face.onnx`: The static-input ONNX model (Works via onnxruntime).
*   `model/face_dataset.txt`: List of images used for quantization.
*   `model/coco_80_labels_list.txt`: Contains only "face".
*   `python/convert.py`: Script used for ONNX -> RKNN conversion (modified).
*   `python/yolo11.py`: Python inference script (modified for face, works with ONNX).
*   `cpp/main.cc`: C++ application entry point (modified).
*   `cpp/rknpu2/yolo11.cc`: C++ RKNN inference logic (modified).
*   `cpp/postprocess.cc`: C++ post-processing logic (rewritten).
*   `cpp/yolo11.h`: C++ header for context struct (modified).
*   `cpp/postprocess.h`: C++ header for post-processing (modified).
*   `cpp/CMakeLists.txt`: Build configuration (modified).
*   `utils/`: Contains helper libraries for image/file operations (used as-is).
