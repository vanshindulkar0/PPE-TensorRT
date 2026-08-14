import time

import cv2
import numpy as np
import tensorrt as trt

from cuda.bindings import runtime

from preprocessing import preprocess_frame
from postprocessing import postprocess


ENGINE_FILE = r"D:\PPE-TensorRT\models\PPE_FINAL_LAST.engine"

logger = trt.Logger(trt.Logger.WARNING)

print("Loading TensorRT engine...")

with open(ENGINE_FILE, "rb") as f:
    engine_data = f.read()

trt_runtime = trt.Runtime(logger)

engine = trt_runtime.deserialize_cuda_engine(
    engine_data
)

if engine is None:
    raise RuntimeError("Failed to load TensorRT engine")

print("TensorRT engine loaded successfully!")



context = engine.create_execution_context()

if context is None:
    raise RuntimeError(
        "Failed to create execution context"
    )

print("Execution context created!")




input_name = None
output_names = []


for i in range(engine.num_io_tensors):

    name = engine.get_tensor_name(i)
    mode = engine.get_tensor_mode(name)

    if mode == trt.TensorIOMode.INPUT:
        input_name = name

    elif mode == trt.TensorIOMode.OUTPUT:
        output_names.append(name)


if input_name is None:
    raise RuntimeError("Input tensor not found")


print("Input:", input_name)
print("Outputs:", output_names)



cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")


print("Webcam opened!")




ret, frame = cap.read()

if not ret:
    raise RuntimeError("Could not read first webcam frame")


original_height, original_width = frame.shape[:2]

input_data = preprocess_frame(frame)


context.set_input_shape(
    input_name,
    input_data.shape
)

outputs = {}

for output_name in output_names:

    output_shape = context.get_tensor_shape(
        output_name
    )

    output_dtype = trt.nptype(
        engine.get_tensor_dtype(output_name)
    )

    outputs[output_name] = np.empty(
        output_shape,
        dtype=output_dtype
    )


err, input_device = runtime.cudaMalloc(
    input_data.nbytes
)

if err != runtime.cudaError_t.cudaSuccess:
    raise RuntimeError(
        f"Could not allocate input GPU memory: {err}"
    )

output_devices = {}

for output_name in output_names:

    output_array = outputs[output_name]

    err, device_ptr = runtime.cudaMalloc(
        output_array.nbytes
    )

    if err != runtime.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"Could not allocate output GPU memory: {err}"
        )

    output_devices[output_name] = device_ptr



err, stream = runtime.cudaStreamCreate()

if err != runtime.cudaError_t.cudaSuccess:
    raise RuntimeError(
        f"Could not create CUDA stream: {err}"
    )

context.set_tensor_address(
    input_name,
    int(input_device)
)

for output_name in output_names:

    context.set_tensor_address(
        output_name,
        int(output_devices[output_name])
    )


print("TensorRT buffers ready!")
print("Starting real-time inference...")
print("Press Q to quit.")



fps = 0.0



while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read webcam frame.")
        break



    original_height, original_width = frame.shape[:2]


    start_time = time.perf_counter()


    input_data = preprocess_frame(frame)


    err, = runtime.cudaMemcpyAsync(
        int(input_device),
        input_data.ctypes.data,
        input_data.nbytes,
        runtime.cudaMemcpyKind.cudaMemcpyHostToDevice,
        stream
    )

    if err != runtime.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"Input copy failed: {err}"
        )


    success = context.execute_async_v3(
        stream_handle=stream
    )

    if not success:
        raise RuntimeError(
            "TensorRT inference failed"
        )

    for output_name in output_names:

        output_array = outputs[output_name]

        err, = runtime.cudaMemcpyAsync(
            output_array.ctypes.data,
            int(output_devices[output_name]),
            output_array.nbytes,
            runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            stream
        )

        if err != runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(
                f"Output copy failed: {err}"
            )


    err, = runtime.cudaStreamSynchronize(stream)

    if err != runtime.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"CUDA synchronization failed: {err}"
        )



    output = outputs[output_names[0]]

    detections = postprocess(
        output,
        original_width,
        original_height,
        confidence_threshold=0.25,
        nms_threshold=0.45
    )

    for detection in detections:

        x, y, w, h = detection["box"]

        class_name = detection["class_name"]

        confidence = detection["confidence"]

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        label = (
            f"{class_name} "
            f"{confidence:.2f}"
        )

        cv2.putText(
            frame,
            label,
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    elapsed = time.perf_counter() - start_time

    if elapsed > 0:
        current_fps = 1.0 / elapsed

     
        fps = (
            0.9 * fps +
            0.1 * current_fps
        )


    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "PPE TensorRT",
        frame
    )


    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()

cv2.destroyAllWindows()


runtime.cudaFree(
    int(input_device)
)

for device_ptr in output_devices.values():

    runtime.cudaFree(
        int(device_ptr)
    )

runtime.cudaStreamDestroy(
    stream
)

print("Webcam inference stopped.")