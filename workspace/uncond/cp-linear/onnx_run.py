import os
import time

import numpy as np
import onnxruntime as ort

# import onnx_converter

# model_path = onnx_converter.output_path

# Load the ONNX model
# model = onnx.load(model_path)
# Check that the IR is well formed
# onnx.checker.check_model(model)
# Print a human readable representation of the graph
# print(onnx.helper.printable_graph(model.graph))

sess = ort.InferenceSession("exp/model.onnx")

input_name = sess.get_inputs()[0].name
input_test = np.ones((1, 1, 7), dtype=np.int32)
# Warmup
pred_onx = sess.run(None, {input_name: input_test})[0]

# Timing
pred_time = time.time()
for i in range(30):
    pred_onx = sess.run(None, {input_name: input_test})[0]
pred_time = time.time() - pred_time
print(pred_time)
