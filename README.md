# Gaussian-Blurring-CUDA
Parallelization and optimization of an image processing application that performs Gaussian blurring using convolution methods in CUDA.

Parallelization of an image processing application that performs Gaussian blurring using convolution methods 
in CUDA (in both block and grid level) and evaluation of the results. I made use of padding technique 
in order to avoid divergence. After the parallelization I optimized the application by making use of the tiling 
method (internal tiling of the image) in order to make use of shared memory. I continued by using multiple kernel 
invocations(external tiling) in order to implement the algorithm in much larger 
image sizes. I made use of CUDA streams in order to perform multiple CUDA operations simultaneously. 
The evaluation was made with the nvvp profiler.
