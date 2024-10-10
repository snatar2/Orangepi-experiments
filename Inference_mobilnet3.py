import numpy as np
import time
from tflite_runtime.interpreter import Interpreter

# Path to the TFLite model file and label map
MODEL_PATH = 'mobilenet_v3_large.tflite'
LABELS_PATH = 'labels.txt'

# Load the TFLite model and allocate tensors
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load the label map
with open(LABELS_PATH, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Prepare input data (example: using random data)
input_shape = input_details[0]['shape']
input_data = np.random.random(input_shape).astype(np.float32)

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], input_data)

# Measure inference time
start_time = time.time()
interpreter.invoke()  # Run inference
end_time = time.time()

# Get the output data
output_data = interpreter.get_tensor(output_details[0]['index'])

# Decode the output
predicted_class = np.argmax(output_data)
class_label = labels[predicted_class]

# Calculate and print inference time and result
inference_time = end_time - start_time
print(f"Inference time: {inference_time:.6f} seconds")
print(f"Predicted class: {predicted_class} - {class_label}")