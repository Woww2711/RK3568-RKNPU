# YOLOv11 Face Detection on RK3568 (Adapted C++ Demo)

## 1. Project Goal

The primary objective is to run a custom-trained YOLOv11n face detection model (`yolov11n-face.pt`, trained on WIDER dataset) efficiently on an RK3568 development board, utilizing its NPU for acceleration. The desired output is a C++ application that takes an image path as input, performs inference using the NPU, and outputs an image with bounding boxes drawn around detected faces.

## 2. Background: Initial Approach & Problem

*   **Initial Attempt:** A Python script using the `rknn.api` library was developed to load a `.rknn` model (converted from the `.pt` file) and run inference directly on the RK3568 board.
*   **Core Problem:** The Python script failed during the NPU initialization step (`rknn.init_runtime()`). Error logs indicated the runtime environment couldn't properly detect or initialize the NPU hardware (`E init_runtime: RKNN model that loaded by 'load_rknn' not support inference on the simulator...`).
*   **Diagnosis:** Troubleshooting revealed issues with the NPU kernel driver (`rknpu`) loading correctly on multiple official vendor-provided Linux OS images (Ubuntu, Debian). Kernel messages (`dmesg`) pointed towards problems related to NPU power management definitions in the Device Tree Blob (DTB), specifically errors like `failed to find power_model node`. Despite DTB inspection showing relevant nodes present, the initialization failed, suggesting a subtle bug or configuration issue within the Board Support Package (BSP).
*   **SDK Success:** Crucially, C++ examples provided within Rockchip's official RKNN SDK (using `librknnrt.so`) *were* able to run successfully on the board, proving the NPU hardware and core libraries were functional.

## 3. Chosen Solution: Adapt C++ Example

Based on the successful execution of the SDK's C++ examples, the most reliable path forward was determined to be adapting an existing C++ example from the `rknn_model_zoo` repository for the face detection task.

The example `rknn_model_zoo/examples/yolo11` (originally for COCO object detection) was chosen as the base.

## 4. Model Conversion Process (Host PC - WSL2)

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
        *   (Recommended) Changed `DEFAULT_RKNN_PATH` to `../model/yolov11n-face.rknn`.
        *   (Recommended for Debugging) Changed `verbose=False` to `verbose=True`.
    *   **Input to Script:** The static-shape `yolov11n-face.onnx`.
    *   **Output:** `yolov11n-face.rknn` (placed in `yolo11/model/`).

## 5. C++ Code Adaptation (`rknn_model_zoo/examples/yolo11/cpp`)

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
    *   `CMakeLists.txt`: Removed conditional compilation logic based on `TARGET_SOC` for other platforms; removed the separate `_zero_copy` build target.
4.  **Post-Processing Rewrite (Single Output Tensor):**
    *   `postprocess.cc`: The main `post_process` function was completely rewritten to handle the model's single output tensor (shape `[1, 5, 8400]`) instead of the original multi-tensor output structure. This involved directly decoding box coordinates (`cx, cy, w, h`) and confidence scores from the single buffer. The old `process_i8`, `process_fp32`, `compute_dfl` functions were removed/integrated.
5.  **Bug Fixes During Adaptation:**
    *   **Type Definitions:** Fixed compilation errors related to conflicting declarations and incomplete types for `rknn_app_context_t` by ensuring `yolo11.h` (where it's defined) was included correctly and removing incorrect forward declarations from `postprocess.h`. Added `typedef struct rknn_app_context_t rknn_app_context_t;` forward declaration in `postprocess.h`.
    *   **Letterbox Coordinates:** Fixed errors in `postprocess.cc` where incorrect member names (`ori_width`/`ori_height`) were used for the `letterbox_t` struct. Identified that original dimensions were missing from `letterbox_t` and modified `post_process` function signature and call site (`inference_yolo11_model` in `yolo11.cc`) to pass original image width/height explicitly.
    *   **RKNN Input Type:** Fixed runtime error `-5` from `rknn_inputs_set` by adding logic in `inference_yolo11_model` (`yolo11.cc`) to correctly detect when the model expects `INT8` input and explicitly converting the `UINT8` image data [0, 255] to the required `INT8` range [-128, 127] before passing the buffer to the RKNN API.
    *   **Padding Color:** Changed `bg_color` in `inference_yolo11_model` (`yolo11.cc`) from `114` to `0` (black) to match Python preprocessing and potentially improve quantized model accuracy.
    *   **Abort Crash:** Mitigated potential `malloc_consolidate` crash on exit by simplifying memory cleanup in `deinit_post_process` (avoiding freeing potentially static label string).

## 6. Compilation

*   **Environment:** Cross-compilation toolchain for `aarch64-linux-gnu` set up (as per RKNN SDK instructions).
*   **Command:** Build script from `rknn_model_zoo` root directory:
    ```bash
    ./build-linux.sh -t rk356x -a aarch64 -d yolo11
    ```
    *(Note: `-t rk356x` covers RK3566/RK3568. `-d yolo11` specifies the example directory name containing the modified `CMakeLists.txt`)*.
*   **Result:** Successful compilation after applying the fixes mentioned above.

## 7. Deployment & Execution

1.  **Locate:** Find the `install/rk356x_linux_aarch64/rknn_yolo11_face_demo/` directory.
2.  **Transfer:** Push the entire `rknn_yolo11_face_demo` directory to the RK3568 board (e.g., `/data/`).
3.  **Push Image:** Copy a test image (e.g., `test.jpg`) to the `model` subdirectory on the board.
4.  **Execute (on board via ADB shell):**
    ```bash
    cd /data/rknn_yolo11_face_demo/
    export LD_LIBRARY_PATH=./lib
    ./rknn_yolo11_face_demo model/yolov11n-face.rknn model/test.jpg
    ```

## 8. Current Status & Known Issues (As of Last Test)

*   **Execution:** The compiled C++ application (`rknn_yolo11_face_demo`) runs successfully on the RK3568 board without crashing during model loading or inference (`rknn_run`).
*   **Issue 1: No Detections:** The primary issue is that **no faces are detected**. Post-processing analysis shows that all potential detections have a dequantized confidence score of `0.0000`. Debug prints revealed the raw INT8 confidence values from the model output buffer are consistently `-128` (the quantization zero-point).
*   **Issue 2: Abort on Exit:** The application sometimes terminates with a `malloc_consolidate(): invalid chunk size Aborted` error, likely related to memory cleanup (potentially the label string freeing). *Update: A fix was applied to `deinit_post_process` to mitigate this.*

## 9. Next Steps: Debugging Quantization

The current evidence (`RawConf=-128`) strongly suggests the problem lies with the **INT8 quantization process during model conversion**. The `.rknn` model itself seems corrupted or poorly calibrated, causing it to output invalid confidence values.

**Debugging focus should return to the conversion stage on the host PC:**

1.  **Verify Conversion Dataset:** Double-check paths and content of `face_dataset.txt`.
2.  **Run Conversion Verbose:** Re-run `python convert.py ... i8 ...` with `RKNN(verbose=True)` and meticulously analyze the build log for warnings or errors related to layer quantization, activation ranges, or fallbacks.
3.  **Test FP32 Model:** Convert the model *without* quantization (`python convert.py ... fp ...`) and run the resulting FP32 `.rknn` model on the board using the C++ demo. If detections work with FP32, it definitively isolates the problem to INT8 quantization.
4.  **Revisit Preprocessing:** Ensure the C++ preprocessing (letterbox, color conversion, UINT8->INT8 conversion) exactly matches the preprocessing assumptions made during the `rknn-toolkit2` conversion (mean/std, color order, data range).
5.  **Consider Toolkit/Model Compatibility:** Investigate if there are known issues with quantizing this specific YOLOv11 variant using `rknn-toolkit2` v2.3.2.

## 10. Project File Overview (Key Files)

*   `model/yolov11n-face.rknn`: The quantized face detection model.
*   `model/face_dataset.txt`: List of images used for quantization.
*   `model/coco_80_labels_list.txt`: Contains only "face".
*   `python/convert.py`: Script used for ONNX -> RKNN conversion (modified).
*   `python/yolo11.py`: Python inference script (modified for face).
*   `cpp/main.cc`: C++ application entry point (modified).
*   `cpp/rknpu2/yolo11.cc`: C++ RKNN inference logic (modified).
*   `cpp/postprocess.cc`: C++ post-processing logic (rewritten).
*   `cpp/yolo11.h`: C++ header for context struct (modified).
*   `cpp/postprocess.h`: C++ header for post-processing (modified).
*   `cpp/CMakeLists.txt`: Build configuration (modified).
*   `utils/`: Contains helper libraries for image/file operations (used as-is).
