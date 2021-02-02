# Transpose matrix multiplication

In this repo you'll find a CUDA implementation of the **A<sup>T</sup> · A** matrix multiplication, where A<sup>T</sup> denotes the transpose of the A matrix.

All three .cu files execute this matrix multiplication, only difference being the speed of it's calculation.

* **parallel-optimized.cu :** in this file the code takes into consideration the GPU architecture and the way the data is organized in it's VRAM in order to minimize memory transfers.
* **parallel-basic.cu :** this file is a naive implementation of the multiplication and serves as a baseline for comparing the speedup achieved through the optimizations.
* **cublas-mm.cu :** this file leverages the cuBLAS library which greatly accelerates the calculation of the most common basic linear algebra operations.
It serves as a benchmark for how fast the multiplication can get.

This code was written in CUDA v8.0 and tested on an Nvidia GTX 970 in Ubuntu 16.04.
