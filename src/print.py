import onnx
from onnx import numpy_helper
import numpy as np
import os

def get_relative_path(file_name:str, dir:str) -> str:
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, dir, file_name)
    return file_path

# Load the .onnx model
model_path = get_relative_path('vae.onnx', 'Models')
model = onnx.load(model_path)

# Get the model's weights
initializers = model.graph.initializer

# Print the names and shapes of the weights
for initializer in initializers:
    print("Name:", initializer.name)
    print("Shape:", initializer.dims)
    # Convert the weights to numpy array and print
    weight_array = numpy_helper.to_array(initializer)
    # weight_array = np.array(initializer)
    print("Weights:", weight_array)
