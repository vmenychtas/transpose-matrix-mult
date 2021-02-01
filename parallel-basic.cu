#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda.h>

void cudaErrorCheck() {
  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    printf("[CUDA error]: %s\n", cudaGetErrorString(code));
    exit(1);
  }
}

__global__ void matrixMultiply(double* mA, double* mC, int rows, int cols){
  int k;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int col = blockIdx.x*blockDim.x + threadIdx.x;

  double val = 0.0;
  if ((row < cols) && (col < cols)){
    for (k = 0; k < rows; k++){
      val += mA[k*cols + row] * mA[k*cols + col];
    }
    mC[row*cols + col] = val;
  }
}

int main(int argc, char *argv[]){
	int i, j, pr_dev = 0;

	if (argv[1] == NULL){
		printf("Matrix dimensions not specified.\n\n");
		exit(1);
	}	else if (argv[2] == NULL) {
		printf("Assuming square matrix.\n\n");
		argv[2] = argv[1];
	}	else if (strcmp(argv[2], "-pd") == 0) {
		printf("Assuming square matrix.\n\n");
		argv[2] = argv[1];
		pr_dev = 1;
	}	else if (argv[3] != NULL) {
		if (strcmp(argv[3], "-pd") == 0) {
			pr_dev = 1;
		}
	}
	int rows = atoi(argv[1]); // A transposed width
	int cols = atoi(argv[2]); // A transposed height -> Dimensions of result C
	/* Allocation and initialization of matrices on host -----------------------*/
	double *hA, *hC;

	hA = (double *)malloc(cols*rows*sizeof(double));
	hC = (double *)malloc(cols*cols*sizeof(double));
	srand(time(NULL));
	for (i = 0; i < rows; i++){
		for (j = 0; j < cols; j++){
			hA[i*cols + j] = rand() / 1000000.0;
		}
	}
	/* Allocation of matrices on the GPU ---------------------------------------*/
	double *dA, *dC;

	cudaMalloc((void **)&dA, cols*rows*sizeof(double));
	cudaMalloc((void **)&dC, cols*cols*sizeof(double));
	cudaMemcpy(dA, hA, rows*cols*sizeof(double), cudaMemcpyHostToDevice);
  /* Get device properties & print them if asked -----------------------------*/
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0);

	if (pr_dev == 1) {
		printf("   [ Device :: %s ]\n", props.name);
		printf("  Number of SMs ---------------: %d\n", props.multiProcessorCount);
		printf("  Global memory ---------------: %lu MB\n", props.totalGlobalMem / 1048576);
		printf("  Constant memory -------------: %lu KB\n", props.totalConstMem / 1024);
		printf("  Threads per warp ------------: %d\n", props.warpSize);
		printf("  Max threads per block -------: %d\n", props.maxThreadsPerBlock);
		printf("  Max registers per block -----: %dK\n", props.regsPerBlock / 1024);
		printf("  Max shared memory per block -: %lu KB\n", props.sharedMemPerBlock / 1024);
		printf("  Max block dimension ---------: %d x %d x %d\n", props.maxThreadsDim[0], props.maxThreadsDim[1], props.maxThreadsDim[2]);
		printf("  Max grid dimension ----------: %d x %d x %d\n\n", props.maxGridSize[0], props.maxGridSize[1], props.maxGridSize[2]);
	}
  /* Kernel invocation & computation timing ----------------------------------*/
  unsigned int block_width = sqrt(props.maxThreadsPerBlock)/2;
  unsigned int grid_dim = (cols - 1) / block_width + 1;
	dim3 dimGrid(grid_dim, grid_dim, 1);
	dim3 dimBlock(block_width, block_width, 1);

	printf("Matrix A ---: %d x %d\n", rows, cols);
	printf("Grid size --: %u x %u\n", grid_dim, grid_dim);
	printf("Block size -: %u x %u\n\n", block_width, block_width);

	cudaEvent_t com_begin, com_end, mem_begin, mem_end;
	cudaEventCreate(&com_begin); cudaEventCreate(&com_end);
	cudaEventCreate(&mem_begin); cudaEventCreate(&mem_end);

	cudaEventRecord(com_begin, 0);
	matrixMultiply<<<dimGrid , dimBlock>>>(dA, dC, rows, cols);
	cudaEventRecord(com_end, 0);
	cudaEventSynchronize(com_end);

	cudaEventRecord(mem_begin, 0);
	cudaMemcpy(hC, dC, cols*cols*sizeof(double), cudaMemcpyDeviceToHost);
	cudaEventRecord(mem_end, 0);
	cudaEventSynchronize(mem_end);

	float com_time = 0;
	float mem_time = 0;
	cudaEventElapsedTime(&com_time, com_begin, com_end);
	cudaEventElapsedTime(&mem_time, mem_begin, mem_end);
	printf("Computation time: %.3f ms\n", com_time);
	printf("Memory transfer: %.3f ms\n", mem_time);
	/* Print output to file for computation evaluation -------------------------*/
	if (cols*rows <= 4096){
		FILE *f = fopen("data.txt", "w");
		if (f == NULL) {
			printf("Error: failed to open file.\n");
			exit(1);
		}

		for (i = 0; i < cols; i++){
			for (j = 0; j < cols; j++){
				fprintf(f, "%f ", hC[i*cols + j]);
			}
			fprintf(f, ";\n");
		}
	}

	free(hA);	cudaFree(dA);
	free(hC);	cudaFree(dC);
	cudaErrorCheck();
	return 0;
}
