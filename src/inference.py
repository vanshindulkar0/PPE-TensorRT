import cv2
import tensorrt as trt
import numpy as np

from cuda.bindings import runtime

from postprocessing import postprocess
from preprocessing import preprocessing

ENGINE_FILE = r"D:\PPE-TensorRT\models\PPE_FINAL_LAST.engine"

IMAGE_FILE = r"D:\PPE-TensorRT\images\ppe_test_image5.jpeg"

print("Initializing CUDA...")
err, count = runtime.cudaGetDeviceCount()

if err != runtime.cudaError_t.cudaSuccess:
    raise RuntimeError(f"CUDA initialization failed: {err}")

print("CUDA initialized successfully!")
print("Number of CUDA devices:", count)

logger = trt.Logger(trt.Logger.INFO)
print("\nLoading TensorRT engine...")

with open(ENGINE_FILE, "rb") as f:
    engine_data = f.read()


runtime_trt = trt.Runtime(logger)

engine = runtime_trt.deserialize_cuda_engine(engine_data)


if engine is None:
    raise RuntimeError("Failed to load TensorRT engine")


print("TensorRT engine loaded successfully!")
context = engine.create_execution_context()


if context is None:
    raise RuntimeError("Failed to create execution context")


print("Execution context created!")

print("\n========== MODEL I/O ==========")

input_name = None
output_names = []


for i in range(engine.num_io_tensors):

    name = engine.get_tensor_name(i)

    shape = engine.get_tensor_shape(name)

    dtype = engine.get_tensor_dtype(name)

    mode = engine.get_tensor_mode(name)

    print(
        f"Name: {name} | "
        f"Shape: {shape} | "
        f"Dtype: {dtype} | "
        f"Mode: {mode}"
    )

    if mode == trt.TensorIOMode.INPUT:

        input_name = name

    elif mode == trt.TensorIOMode.OUTPUT:

        output_names.append(name)


if input_name is None:
    raise RuntimeError("Could not find model input")


print("\nTensorRT input:", input_name)

print("TensorRT outputs:", output_names)

original_image = cv2.imread(IMAGE_FILE)

if original_image is None:
    raise FileNotFoundError(
        f"Could not read image: {IMAGE_FILE}"
    )

original_height, original_width = original_image.shape[:2]

print(
    f"Original image size: "
    f"{original_width}x{original_height}"
)


print("\n========== PREPROCESSING ==========")

input_data = preprocessing(IMAGE_FILE)

print("\nInput ready for TensorRT!")

print("Input shape:", input_data.shape)

print("Input dtype:", input_data.dtype)

context.set_input_shape(
    input_name,
    input_data.shape
)

print("\nInput shape set successfully!")

outputs = {}

print("\n========== OUTPUT INFORMATION ==========")


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

    print(
        f"{output_name}: "
        f"shape={output_shape}, "
        f"dtype={output_dtype}"
    )
print("\n========== GPU MEMORY ==========")

input_size = input_data.nbytes


err, input_device = runtime.cudaMalloc(input_size)


if err != runtime.cudaError_t.cudaSuccess:

    raise RuntimeError(
        f"Failed to allocate input GPU memory: {err}"
    )


print(
    f"Input GPU memory allocated: "
    f"{input_size / (1024 * 1024):.2f} MB"
)

output_devices = {}


for output_name in output_names:

    output_array = outputs[output_name]

    output_size = output_array.nbytes

    err, output_device = runtime.cudaMalloc(
        output_size
    )


    if err != runtime.cudaError_t.cudaSuccess:

        raise RuntimeError(
            f"Failed to allocate GPU memory "
            f"for {output_name}: {err}"
        )


    output_devices[output_name] = output_device

    print(
        f"{output_name} GPU memory allocated: "
        f"{output_size / (1024 * 1024):.2f} MB"
    )

err, stream = runtime.cudaStreamCreate()


if err != runtime.cudaError_t.cudaSuccess:

    raise RuntimeError(
        f"Failed to create CUDA stream: {err}"
    )


print("CUDA stream created!")

context.set_tensor_address(
    input_name,
    input_device
)


for output_name in output_names:

    context.set_tensor_address(
        output_name,
        output_devices[output_name]
    )


print("Tensor addresses assigned!")

print("\n========== HOST → DEVICE ==========")

(err,) = runtime.cudaMemcpyAsync(
    input_device,
    input_data.ctypes.data,
    input_data.nbytes,
    runtime.cudaMemcpyKind.cudaMemcpyHostToDevice,
    stream
)

if err != runtime.cudaError_t.cudaSuccess:

    raise RuntimeError(
        f"Failed to copy input CPU → GPU: {err}"
    )


print("Input copied from CPU → GPU!")


print("\n========== TENSORRT INFERENCE ==========")

success = context.execute_async_v3(
    stream_handle=stream
)


if not success:

    raise RuntimeError(
        "TensorRT inference failed!"
    )


print("TensorRT inference executed successfully!")


print("\n========== DEVICE → HOST ==========")


for output_name in output_names:

    output_array = outputs[output_name]

    output_device = output_devices[output_name]


    (err,) = runtime.cudaMemcpyAsync(
    output_array.ctypes.data,
    output_device,
    output_array.nbytes,
    runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
    stream
)


    if err != runtime.cudaError_t.cudaSuccess:

        raise RuntimeError(
            f"Failed to copy {output_name} "
            f"GPU → CPU: {err}"
        )

(err,) = runtime.cudaStreamSynchronize(stream)


if err != runtime.cudaError_t.cudaSuccess:

    raise RuntimeError(
        f"CUDA synchronization failed: {err}"
    )
    
print("\n========== OUTPUT STATISTICS ==========")

output = outputs["output0"]

print("\n========== POSTPROCESSING ==========")

detections = postprocess(
    output,
    original_width,
    original_height,
    confidence_threshold=0.25,
    nms_threshold=0.45,
)

print(f"Detections found: {len(detections)}")

for detection in detections:

    print(
        f"{detection['class_name']} "
        f"{detection['confidence']:.2f} "
        f"Box: {detection['box']}"
    )
    
    
for detection in detections:

    x, y, w, h = detection["box"]

    class_name = detection["class_name"]
    confidence = detection["confidence"]

    cv2.rectangle(
        original_image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2,
    )

    label = f"{class_name} {confidence:.2f}"

    cv2.putText(
        original_image,
        label,
        (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    
output_path = r"D:\PPE-TensorRT\output\detection.jpg"

cv2.imwrite(
    output_path,
    original_image
)

cv2.imshow("Detections", original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Result saved to: {output_path}")
print("Shape:", output.shape)
print("Min:", output.min())
print("Max:", output.max())
print("Mean:", output.mean())

print("\nFirst prediction:")
print(output[0, :, 0])


print("Output copied from GPU → CPU!")

print("\n========== RAW MODEL OUTPUT ==========")


for output_name in output_names:

    output = outputs[output_name]

    print(
        f"\nOutput: {output_name}"
    )

    print(
        "Shape:",
        output.shape
    )

    print(
        "Dtype:",
        output.dtype
    )
    print("Output shape:", output.shape)
    print("\nFirst prediction:")
    print(output[0, :, 0])
    print("\nSecond prediction:")
    print(output[0, :, 1])
    print("\nThird prediction:")
    print(output[0, :, 2])


runtime.cudaFree(input_device)


for output_device in output_devices.values():

    runtime.cudaFree(output_device)


runtime.cudaStreamDestroy(stream)


print("\nGPU memory released!")

print("\n========== INFERENCE COMPLETE ==========")
