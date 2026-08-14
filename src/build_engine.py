import tensorrt as trt

model_path="D:\\PPE-TensorRT\\models\\PPE_FINAL_LAST.onnx"
engine_path="D:\\PPE-TensorRT\\models\\PPE_FINAL_LAST.engine"

logger=trt.Logger(trt.Logger.INFO)

builder=trt.Builder(logger)
network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

parser=trt.OnnxParser(network,logger)

with open(model_path,'rb') as model:
    if not parser.parse(model.read()):
        print('failed to load ONNX file')
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        
        raise Exception('Failed to load ONNX file')

print("the model has been loaded successfully")

config=builder.create_builder_config()

if builder.platform_has_fast_fp16:
    config.set_flag(trt.BuilderFlag.FP16)
    print("FP16 mode is enabled")

print("Building TensorRT engine...")

serialized_engine = builder.build_serialized_network(
    network,
    config
)

if serialized_engine is None:
    raise RuntimeError("Failed to build TensorRT engine")


with open(engine_path, "wb") as f:
    f.write(serialized_engine)


print("TensorRT engine created successfully!")
print(f"Saved to: {engine_path}")