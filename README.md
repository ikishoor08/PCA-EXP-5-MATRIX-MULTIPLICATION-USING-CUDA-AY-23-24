# PCA-EXP-5-MATRIX-MULTIPLICATION-USING-CUDA-AY-23-24
<h3>AIM:</h3>
<h3>ENTER YOUR NAME: KISHOOR I</h3>
<h3>ENTER YOUR REGISTER NO: 212225040190</h3>
<h3>EX. NO: 5</h3>
<h1> <align=center> MATRIX MULTIPLICATION USING CUDA </h3>
  Implement Matrix Multiplication using GPU.</h3>

## AIM:
To perform Matrix Multiplication using CUDA and check its performance with nvprof.
## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
## PROCEDURE:
1.	Define Constants: Define the size of the matrices (SIZE) and the size of the CUDA blocks (BLOCK_SIZE).
2.	Kernel Function: Define a CUDA kernel function matrixMultiply that performs the matrix multiplication.
3.	In the main function, perform the following steps:
4.	Initialize Matrices: Initialize the input matrices ‘a’ and ‘b’ with some values.
5.	Allocate Device Memory: Allocate memory on the GPU for the input matrices ‘a’ and ‘b’, and the output matrix ‘c’.
6.	Copy Matrices to Device: Copy the input matrices from host (CPU) memory to device (GPU) memory.
7.	Set Grid and Block Sizes: Set the grid and block sizes for the CUDA kernel launch.
8.	Start Timer: Start a timer to measure the execution time of the kernel.
9.	Launch Kernel: Launch the matrixMultiply kernel with the appropriate grid and block sizes, and the input and output matrices as arguments.
10.	Copy Result to Host: After the kernel execution, copy the result matrix from device memory to host memory.
11.	Stop Timer: Stop the timer and calculate the elapsed time.
12.	Print Result: Print the result matrix and the elapsed time.
13.	Free Device Memory: Finally, free the device memory that was allocated for the matrices.
## PROGRAM:
```

%%writefile matmul.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define SIZE 4
#define BLOCK_SIZE 2

__global__ void matrixMultiply(int *a, int *b, int *c, int size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size)
    {
        int sum = 0;
        for (int k = 0; k < size; k++)
        {
            sum += a[row * size + k] * b[k * size + col];
        }
        c[row * size + col] = sum;
    }
}

int main()
{
    int a[SIZE][SIZE], b[SIZE][SIZE], c[SIZE][SIZE];
    int *d_a, *d_b, *d_c;

    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            a[i][j] = i + j;
            b[i][j] = i - j;
        }
    }

    cudaMalloc((void**)&d_a, SIZE * SIZE * sizeof(int));
    cudaMalloc((void**)&d_b, SIZE * SIZE * sizeof(int));
    cudaMalloc((void**)&d_c, SIZE * SIZE * sizeof(int));

    cudaMemcpy(d_a, a, SIZE * SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, SIZE * SIZE * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(SIZE / BLOCK_SIZE, SIZE / BLOCK_SIZE);

    struct timeval start, end;
    gettimeofday(&start, NULL);

    matrixMultiply<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, SIZE);
    cudaDeviceSynchronize();

    gettimeofday(&end, NULL);

    cudaMemcpy(c, d_c, SIZE * SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Result Matrix:\n");
    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            printf("%d ", c[i][j]);
        }
        printf("\n");
    }

    double elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_usec - start.tv_usec) / 1000000.0;

    printf("Elapsed Time: %f seconds\n", elapsed);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
```

## OUTPUT:


<img width="572" height="172" alt="WhatsApp Image 2026-11-23 at 2 25 53 PM" src="https://github.com/user-attachments/assets/f34ad9f9-eb77-4d62-92d7-7d37ab9a1806" />


## RESULT:
The matrix multiplication operation was successfully implemented using CUDA. The program executed on the NVIDIA Tesla T4 GPU and produced the correct result matrix with an elapsed time of 0.000732 seconds.
