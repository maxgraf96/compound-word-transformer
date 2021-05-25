import os
import pickle

import torch
import main_cp

output_path = "exp/model.onnx"


def convert(output_path):
    # Load net
    path_ckpt = main_cp.info_load_model[0]  # path to ckpt dir
    loss = main_cp.info_load_model[1]  # loss
    name = 'loss_' + str(loss)
    path_saved_ckpt = os.path.join(path_ckpt, name + '_params.pt')

    # load
    dictionary = pickle.load(open(main_cp.path_dictionary, 'rb'))
    event2word, word2event = dictionary

    # config
    n_class = []
    for key in event2word.keys():
        n_class.append(len(dictionary[0][key]))

    # init model
    net = main_cp.TransformerModel(n_class, is_training=False)
    net.cuda()
    net.eval()

    # load model
    print('[*] load model from:', path_saved_ckpt)
    net.load_state_dict(torch.load(path_saved_ckpt))

    print("Converting to ONNX format.")

    input = torch.ones((1, 1, 7), dtype=torch.int32).cuda()
    torch.onnx.export(
        net,
        input,
        output_path,
        verbose=True,
        input_names=['input'],
        output_names=['output'],
        opset_version=12
    )
    print("Done converting, saved to", output_path)


def convert_onnx_to_trt():
    ONNX_FILE_PATH = 'exp/model.onnx'

    import onnx
    onnx_model = onnx.load(ONNX_FILE_PATH)
    onnx.checker.check_model(onnx_model)

    import pycuda.driver as cuda
    import numpy as np
    import tensorrt as trt

    # logger to capture errors, warnings, and other information during the build and inference phases
    TRT_LOGGER = trt.Logger()

    def build_engine(onnx_file_path):
        # initialize TensorRT engine and parse ONNX model
        builder = trt.Builder(TRT_LOGGER)

        network_creation_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION)
        network_creation_flag |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_creation_flag)
        config = builder.create_builder_config()
        parser = trt.OnnxParser(network, TRT_LOGGER)

        # parse ONNX
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            model_parsed_correctly = parser.parse(model.read())
            support = parser.supports_model(model.read())
            error1 = parser.get_error(0)
            error2 = parser.get_error(1)
            a = 2
        print('Completed parsing of ONNX file')

        # we have only one image in batch
        builder.max_batch_size = 1
        # use FP16 mode if possible
        # if builder.platform_has_fast_fp16:
        #     builder.fp16_mode = True

        print('Building an engine...')
        engine = builder.build_engine(network, config)
        context = engine.create_execution_context()
        print("Completed creating Engine")

        return engine, context

    # initialize TensorRT engine and parse ONNX model
    engine, context = build_engine(ONNX_FILE_PATH)

    # get sizes of input and output and allocate memory required for input data and for output data
    for binding in engine:
        if engine.binding_is_input(binding):  # we expect only one input
            input_shape = engine.get_binding_shape(binding)
            input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize  # in bytes
            device_input = cuda.mem_alloc(input_size)
        else:  # and one output
            output_shape = engine.get_binding_shape(binding)
            # create page-locked memory buffers (i.e. won't be swapped to disk)
            host_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float32)
            device_output = cuda.mem_alloc(host_output.nbytes)

        # Create a stream in which to copy inputs/outputs and run inference.
        stream = cuda.Stream()

        cuda.memcpy_htod_async(device_input, input, stream)

        context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_output, device_output, stream)
        stream.synchronize()

        output_data = torch.Tensor(host_output).reshape(engine.max_batch_size, output_shape[0])
        print(output_data)


if __name__ == '__main__':
    convert(output_path)
    convert_onnx_to_trt()