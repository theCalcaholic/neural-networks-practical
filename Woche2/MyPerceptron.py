import numpy
import KTimage as KT

# directory for observing the results
path_obs = "res/"

# XOR data
data_in = numpy.array([[0,0],[0,1],[1,0],[1,1]])
data_out = numpy.array([[0],[1],[1],[0]])

# size of layer 0 (input layer) is set to the size of a data point
size_0 = numpy.shape(data_in)[1]

# size of layer 1 (output layer)
size_1 = 1

# allocate weight matrix from layer 0 to layer 1
W_1_0 = numpy.random.uniform(-1.0, 1.0, (size_1, size_0))

# feedforward activation
S_1 = numpy.dot(W_1_0, data_in[2])

# write out the activation vectors and weights as images for display with look.py
# note: file names must follow convention (using "obs", "_", letter, and digit(s)) to interplay with look.py
KT.exporttiles (data_in[2], size_0, 1, path_obs+"obs_S_0.pgm")
KT.exporttiles (S_1, size_1, 1, path_obs+"obs_S_1.pgm")
KT.exporttiles (W_1_0, size_0, 1, path_obs+"obs_W_1_0.pgm", size_1, 1)

