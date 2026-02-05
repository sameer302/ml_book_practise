import numpy as np
import timeit # timeit module provides a simple way to time small bits of Python code

def forloop(x,w):
    z = 0
    for i in range(len(x)):
        z += x[i]*w[i]
    return z

def listcomprehension(x,w):
    return sum(x_i*w_i for x_i, w_i in zip(x,w))

def vectorized(x,w):
    return x.dot(w) # list object has no attribute 'dot', it is only for numpy arays


x, w = np.random.rand(100000), np.random.rand(100000) # generates random numbers from a uniform distribution over the interval [0,1)


print(timeit.timeit(lambda: forloop(x, w), number=10))
''' 
timeit.timeit(statement, number=N), here statement is the code that we want to run and number is how many times the code is executed and the return value is in seconds.

`timeit` needs a function it can call repeatedly, not a function that runs immediately. When you write `forloop(x, w)`, Python executes it right away and passes its result 
(or `None` if there's no return) to `timeit`, so there is nothing left to measure. Removing the `return` does not help because the function still runs immediately. 
Using `lambda: forloop(x, w)` works because it creates a temporary function (like `def temp(): forloop(x, w)`) that does not run right away; instead, `timeit` can call this 
temporary function many times to correctly measure how long the code takes to run.
'''
print(timeit.timeit(lambda: listcomprehension(x, w), number=10))
print(timeit.timeit(lambda: vectorized(x, w), number=10))

'''
NumPy uses CPU optimized math instructions hence it is faster than the other two ways. Even though NumPy and Python for-loops run on the same CPU, cache, and RAM, the speed 
difference happens because NumPy uses the hardware much more efficiently. Python loops are slow due to interpreter overhead, repeated type checking, and object handling, 
while NumPy performs operations in optimized compiled C/ FORTRAN code, uses vectorized CPU instructions (SIMD), accesses memory more efficiently with contiguous arrays, and relies on highly 
optimized math libraries (Basic Linear Algebra Subprograms/BLAS, Linear Algebra Package). The overall speed gain is not due to one factor but the combined effect of multiple low-level optimizations working together.

By default, normal Python and NumPy code always run on the CPU, and GPU is used only if you explicitly use GPU-enabled libraries like PyTorch, TensorFlow, JAX, or CuPy and 
move data/models to the GPU. You can check CPU usage using `top` or `htop`, and GPU usage (for NVIDIA GPUs) using `nvidia-smi`, which shows active processes and memory usage. 
Inside code, PyTorch provides `torch.cuda.is_available()` and tensor `.device` to confirm execution device, while TensorFlow lists GPUs using `tf.config.list_physical_devices('GPU')`. 
If you do not explicitly specify GPU usage (for example by using `"cuda"` in PyTorch), your code is running on the CPU.


'''

# One thing is to look into what different techniques are being used by numpy to optimize the operations.
# Other thing is the effect of cache warmup on the timings. Running the same code multiple times might lead to better cache utilization and hence better timings.
# One more thing is to take multiple values and plot them to see are the observations aligned with what we hypothesize or not. 