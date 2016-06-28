import numpy as np
from util import Logger
import KTimage
import os
import pyopencl as cl

OPENCL_ENABLED = True

class NeuralLayer(object):
    """Simple perceptron layer with weight matrix"""
    def __init__(self, in_size, out_size, activation_fn=None, activation_fn_deriv=None):
        Logger.debug("create Neural Layer: \n" + \
                          "  in_size: " + str(in_size) + "\n" + \
                          "  out_size: " + str(out_size))
        # initialize weight matrix
        self.weights = np.random.uniform(-0.001, 0.001, (out_size, in_size)).astype(np.float32)
        # set biases if not already set (by child class for example)
        if not hasattr(self, 'biases'):
            self.biases = np.zeros(out_size).astype(np.float32)
        # set activation function and derivate
        self.activation = activation_fn or NeuralLayer.activation_linear
        self.activation_deriv = activation_fn_deriv or NeuralLayer.activation_linear_deriv
        # set size and input size
        self.size = out_size
        self.in_size = in_size
        if OPENCL_ENABLED:
            print(cl.get_cl_header_version())
            self.ctx = cl.create_some_context(True)
            self.queue = cl.CommandQueue(self.ctx)

    def feed(self, input_data):
        # calculate activation of layer for given inputs
        #Logger.DEBUG = True
        Logger.debug("NeuralLayer:feed")
        Logger.debug("input: " + str(np.shape(input_data)))
        Logger.debug("biases: " + str(np.shape(self.biases)))
        input_data = input_data.astype(np.float32)
        dot = np.dot(self.weights, input_data).flatten()
        Logger.debug("dot: " + str(np.shape(dot)))
        if Logger.DEBUG:
            raw_input("...")
        if OPENCL_ENABLED:
            mf = cl.mem_flags
            cl_dot = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dot)
            cl_biases = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.biases)
            prg = cl.Program(self.ctx, """
            __kernel void sum( __global const float *a_g, __global const float *b_g, __global float *res_g ) {
                int gid = get_global_id(0);
                res_g[gid] += a_g[gid] + b_g[gid];
            }
            """).build()
            cl_result = cl.Buffer(self.ctx, mf.READ_WRITE, dot.nbytes)
            prg.sum(self.queue, dot.shape, None, cl_dot, cl_biases, cl_result)
            addition = np.empty_like(dot)
            cl.enqueue_copy(self.queue, addition, cl_result)
        else:
            addition = dot + self.biases.T
        result = self.activation(addition)
        Logger.debug("output: " + str(np.shape(result)))
        #Logger.debug("weights: " + str(np.shape(self.weights)))
        #Logger.debug("dot shape: " + str(np.shape(dot)))
        Logger.DEBUG = False
        return result

    def learn(self, result, delta, learning_rate):
        #raw_input("press Enter")
        # apply learning rule
        #Logger.DEBUG = True
        Logger.debug("NeuralLayer:learn")
        Logger.debug("result: " + str(np.shape(result)))
        Logger.debug("delta: " + str(np.shape(delta)))# + "\nresult shape:" + str(np.shape(result)))
        delta_weights = learning_rate * np.outer(delta, result)
        #Logger.debug("delta weights shape:" + str(np.shape(delta_weights)))
        #Logger.log(str(delta_weights))
        self.weights += delta_weights
        #Logger.DEBUG = False

    def get_delta(self, out_data, last_delta, last_weights):
        """calculate delta for layer before"""
        #Logger.DEBUG = True
        Logger.debug("Get delta: ")
        Logger.debug("out: " + str(np.shape(out_data)))
        Logger.debug("last_delta: " + str(np.shape(last_delta)))
        Logger.debug("last_weights: " + str(np.shape(last_weights)))
        dot = np.dot(last_weights.T, last_delta)
        Logger.debug("dot shape: " + str(np.shape(dot)))
        delta = dot * self.activation_deriv(out_data)
        #Logger.debug("delta shape: " + str(np.shape(delta)))
        #Logger.DEBUG = False
        return delta

    def save(self, path):
        """save weights and biases to path"""
        if not os.path.isdir(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        np.savez(path, weights=self.weights, biases=self.biases)

    def load(self, path):
        """load weights and biases from path"""
        np_saved = np.load(path)
        self.weights = np_saved['weights']
        self.biases = np_saved['biases']

    def visualize(self, path):
        """create visualization for weights at <path>"""
        if not os.path.isdir(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        KTimage.exporttiles(self.weights, self.in_size,
                            1, path, 1, self.size)

    @classmethod
    def activation_linear(cls, x):
        return x

    @classmethod
    def activation_linear_deriv(cls, x):
        return 1

    @classmethod
    def activation_sigmoid(cls, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    @classmethod
    def activation_sigmoid_deriv(cls, x):
        return (1. - x) * x

    @classmethod
    def activation_tanh(cls, x):
        return np.tanh(x)

    @classmethod
    def activation_tanh_deriv(cls, x):
        return 1.0 - x ** 2


class BiasedNeuralLayer(NeuralLayer):
    """Neural Layer with biases"""
    def learn(self, result, delta, learning_rate):
        super(BiasedNeuralLayer, self).learn(result, delta, learning_rate)
        Logger.debug("delta: " + str(delta))
        Logger.debug("biases: " + str(self.biases))
        tmp = -(learning_rate * delta)
        self.biases = tmp + np.atleast_2d(self.biases)
        self.biases[self.biases < 0] = 0
        #self.biases = min(0, tmp + np.atleast_2d(self.biases))
        #self.biases += learning_rate * delta
