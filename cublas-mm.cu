/*
 Compilation: nvcc -lcublas
 Execution: sudo ldconfig /usr/local/cuda-8.0/lib64
*/
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "cublas_v2.h"

int main(int argc, char *argv[]){
  int i, j;

  if (argv[1] == NULL){
    printf("Matrix dimensions not specified.\n\n");
    exit(1);
  } else if (argv[2] == NULL) {
    printf("Assuming square matrix.\n\n");
    argv[2] = argv[1];
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

  cudaMalloc((void **) &dA, cols*rows*sizeof(double));
  cudaMalloc((void **) &dC, cols*cols*sizeof(double));
  cublasSetMatrix(rows, cols, sizeof(double), hA, rows, dA, rows);
  /* CUBLAS invocation & computation timing ----------------------------------*/
  double alpha = 1.0;
  double beta = 0.0;
  cublasOperation_t opN = CUBLAS_OP_N;
  cublasOperation_t opT = CUBLAS_OP_T;
  cublasStatus_t status;
  cublasHandle_t handle;
  cublasCreate(&handle);

  cudaEvent_t cub_begin, cub_end, mem_begin, mem_end;
  cudaEventCreate(&cub_begin); cudaEventCreate(&cub_end);
  cudaEventCreate(&mem_begin); cudaEventCreate(&mem_end);

  cudaEventRecord(cub_begin, 0);
  status = cublasDgemm(handle,opN,opT,cols,cols,rows,&alpha,dA,cols,dA,cols,&beta,dC,cols);
  cudaEventRecord(cub_end, 0);
  cudaEventSynchronize(cub_end);

  if (status == CUBLAS_STATUS_SUCCESS){
    cudaEventRecord(mem_begin, 0);
    cudaMemcpy(hC, dC, cols*cols*sizeof(double), cudaMemcpyDeviceToHost);
    cudaEventRecord(mem_end, 0);
  	cudaEventSynchronize(mem_end);

    float cub_time = 0;
    float mem_time = 0;
    cudaEventElapsedTime(&cub_time, cub_begin, cub_end);
    cudaEventElapsedTime(&mem_time, mem_begin, mem_end);
    printf("Computation time: %.3f ms\n", cub_time);
    printf("Memory transfer: %.3f ms\n", mem_time);
  } else {
    printf("Error: cublasDgemm failed to compute.\n");
    exit(1);
  }
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

  free(hA); cudaFree(dA);
  free(hC); cudaFree(dC);
  return 0;
}
