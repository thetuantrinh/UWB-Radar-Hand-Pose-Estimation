
import numpy as np
import time
# Use autoprimaryctx if available (pycuda >= 2021.1) to
# prevent issues with other modules that rely on the primary
# device context.
try:
    import pycuda.autoprimaryctx
except ModuleNotFoundError:
    import pycuda.autoinit

import pycuda.driver as cuda
import tensorrt as trt

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


def GiB(val):
    return int(val * (1 << 30))


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        print(f"data type of the bulit engine is {dtype}")
        print(f"size {size} batch {engine.max_batch_size}")
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

# This function is generalized for multiple inputs/outputs for full dimension networks.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

def do_inference_v3(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    st = time.time()
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    et1 = time.time()
    data_tran_time = et1 - st
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    et2 = time.time()
    latency = et2 - et1
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    et3 = time.time()
    back = et3 - et2
    # Synchronize the stream
    stream.synchronize()
    total = et3 - st
    # Return only the host outputs.
    return data_tran_time, latency, back, total


# the following two functions are based on "https://github.com/NVIDIA/TensorRT/blob/main/quickstart/IntroNotebooks/4.%20Using%20PyTorch%20through%20ONNX.ipynb"

def allocate_mem(input_batch,
                 batch_size=32,
                 output_classes=12,
                 target_dtype=np.float16
                 ):
    output = np.empty([batch_size, output_classes], dtype=target_dtype) # Need to set output dtype to FP16 to enable FP16

    # Allocate device memory
    d_input = cuda.mem_alloc(1 * input_batch.nbytes)
    d_output = cuda.mem_alloc(1 * output.nbytes)

    bindings = [int(d_input), int(d_output)]
    stream = cuda.Stream()
    return d_input, stream, bindings, d_output, output

def validate(X_valid,
             Y_valid,
             d_input,
             d_output,
             bindings,
             stream,
             context,
             output,
             batch_size
             ):
    
    n_batch = int(np.ceil(Y_valid.shape[0]/batch_size))
    total = Y_valid.shape[0]
    correct = 0

    for batch in range(n_batch):
        input_batch = X_valid[batch * batch_size:((batch + 1) * batch_size), :]
        groundtruth = Y_valid[batch * batch_size:((batch + 1) * batch_size), :]
        
        # Transfer input data to device
        cuda.memcpy_htod_async(d_input, input_batch, stream)
        # Execute model
        context.execute_async_v2(bindings, stream.handle, None)
        # Transfer predictions back
        cuda.memcpy_dtoh_async(output, d_output, stream)
        # Syncronize threads
        stream.synchronize()

        if groundtruth.shape[0] != output.shape[0]:
            output = output[0:groundtruth.shape[0], :]

        guessed = np.argmax(output, axis=1)
        expected = np.argmax(groundtruth, axis=1)
        correct += np.sum(guessed == expected)

    return 100*(correct/total)


def predict(X_batch,
            d_input,
            d_output,
            bindings,
            stream,
            context,
            output,
            batch_size
            ):
    
    st = time.time()
    # Transfer input data to device
    cuda.memcpy_htod_async(d_input, X_batch, stream)
    et = time.time()
    data_tran_time = et - st
    # Execute model
    context.execute_async_v2(bindings, stream.handle, None)
    # Transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)
    # Syncronize threads
    stream.synchronize()
    et1 = time.time()
    latency = et1 - et
    runtime = et1 - st

    return data_tran_time, latency, runtime, output


# Tensorflow eval function
def eval(model, X_test, Y_test, batch_size):

    n_batch = int(np.ceil(Y_test.shape[0]/batch_size))
    total = Y_test.shape[0]
    correct = 0
    idx = 0

    for batch in range(n_batch):
        val_batch = X_test[batch * batch_size:((batch + 1) * batch_size), :]
        preds = model.predict(val_batch)
        # groundtruth = Y_test[batch:(batch + batch_size), :]
        groundtruth = Y_test[batch * batch_size:((batch + 1) * batch_size), :]

        guessed = np.argmax(preds, axis=1)
        expected = np.argmax(groundtruth, axis=1)
        correct += (guessed == expected).sum()
        
    return 100*(correct/total)    
