import numpy as np
import pyopencl as cl

in_size = 2
out_size = 3

input_data = np.array([[0], [1]]).astype(np.float32)

weights = np.random.uniform(-0.001, 0.001, (out_size, in_size)).astype(np.float32);
biases = np.zeros(out_size).astype(np.float32)

a_np = np.random.rand(50000).astype(np.float32)
b_np = np.random.rand(50000).astype(np.float32)
#biases = np.random.rand(3).astype(np.float32)

dot = np.dot(weights, input_data).flatten()
#dot = np.random.rand(3).astype(np.float32)

print("biases: " + str(np.shape(biases)) + "(" + str(biases) + ")" + "[" + str(type(biases)) + "]")
print("dot: " + str(np.shape(dot)) + "(" + str(dot) + ")")


ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)


# calculation

mf = cl.mem_flags
cl_dot = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dot)
cl_biases = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=biases)


a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)
#prg = cl.Program(ctx, """
#__kernel void sum( __global const float *a_g, __global const float *b_g, __global float *res_g ) {
#    int gid = get_global_id(0);
#    res_g[0][gid] = a_g[0][gid] + b_g[0][gid];
#}
#""").build()

prg = cl.Program(ctx, """
__kernel void sum(
    __global const float *a_g, __global const float *b_g, __global float *res_g)
{
  int gid = get_global_id(0);
  res_g[gid] = a_g[gid] + b_g[gid];
}
""").build()

cl_result = cl.Buffer(ctx, mf.WRITE_ONLY, dot.nbytes)
prg.sum(queue, dot.shape, None, cl_dot, cl_biases, cl_result)
res_np = np.empty_like(dot)
cl.enqueue_copy(queue, res_np, cl_result)

res = dot + biases


"""
res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
prg.sum(queue, a_np.shape, None, a_g, b_g, res_g)

res_np = np.empty_like(a_np)
cl.enqueue_copy(queue, res_np, res_g)

res = a_np + b_np
"""

print("### sum ###")
print(str(res))
print("### cl sum ###")
print(str(res_np))
