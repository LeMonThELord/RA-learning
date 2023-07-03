#include "helper.h"
#include "tag.h"
#include <iostream>

#define DEBUG (0)

void cudaMatrixAlloc(void **devPtr, size_t size)
{
    cudaError_t status;

    status = cudaMalloc(devPtr, size);

    if (status != cudaSuccess)
    {
        cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;
    }
}

__global__ void matrixMultiplication(float *A, float *B, float *R, int R_rows, int M_dims, int R_cols)
{

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < R_rows && col < R_cols)
    {
        float temp = 0;
        for (int i = 0; i < M_dims; i++)
        {
            temp += A[row * M_dims + i] * B[i * R_cols + col];
        }

        R[row * R_cols + col] = temp;
    }
}

int main(int argc, char **argv)
{

    // Setup

    int min_dim, max_dim, step; // args variables

    if (argc < 4 || argc > 5)
    {
        printf("Usage: %s <min_dim> <max_dim> <step>\n", argv[0]);
        return 1;
    }

    min_dim = atoi(argv[1]);
    max_dim = atoi(argv[2]);
    step = atoi(argv[3]);

    // Hardware info
    cudaDeviceProp gpu_prop;
    cudaGetDeviceProperties(&gpu_prop, 0);
    int MaxThreadsPerBlock = gpu_prop.maxThreadsPerBlock;

    // Calc and perf measure
    float *A, *B, *R;
    float *gpu_A, *gpu_B, *gpu_R;
    tag_point start, end;
    tag_segment load_duration, calc_duration, read_duration;

    cudaEvent_t begin, stop;
    cudaEventCreate(&begin);
    cudaEventCreate(&stop);
    float event_time_in_millis;

    for (int dim = min_dim; dim <= max_dim; dim += step)
    {
        A = matrix2ArrayMatrix(createRandomMatrix(dim, dim), dim, dim);
        B = matrix2ArrayMatrix(createRandomMatrix(dim, dim), dim, dim);
        R = matrix2ArrayMatrix(createZerosMatrix(dim, dim), dim, dim);

        if (DEBUG)
        {
            pprintArrayMatrix(A, dim, dim, "pthread A");
            pprintArrayMatrix(B, dim, dim, "pthread B");
        }

        long data_size = dim * dim * sizeof(float);

        dim3 threadsPerBlock(dim, dim);
        dim3 blocksPerGrid(1, 1);

        if (dim * dim > MaxThreadsPerBlock)
        {
            threadsPerBlock.x = MaxThreadsPerBlock;
            threadsPerBlock.y = MaxThreadsPerBlock;
            blocksPerGrid.x = ceil(float(dim) / float(threadsPerBlock.x));
            blocksPerGrid.y = ceil(float(dim) / float(threadsPerBlock.y));
        }

        // allocate memory to contain the matrices
        cudaMatrixAlloc((void **)&gpu_A, data_size);
        cudaMatrixAlloc((void **)&gpu_B, data_size);
        cudaMatrixAlloc((void **)&gpu_R, data_size);

        start = tagTime();

        //----------------------LOAD PART----------------------

        cudaMemcpy(gpu_A, A, data_size, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_B, B, data_size, cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();

        //----------------------LOAD PART----------------------

        end = tagTime();

        load_duration = end - start;

        // Write the time in the csv file
        saveTimeToFile(dim, load_duration.count(), "csv/load-cuda.csv");

        cudaEventRecord(begin, 0); // begin "recording" operations on GPU
        start = tagTime();
        //----------------------MATMULT PART----------------------
        // load and execute the kernel to multiplication into the GPU

        matrixMultiplication<<<blocksPerGrid, threadsPerBlock>>>(gpu_A, gpu_B, gpu_R, dim, dim, dim);
        cudaDeviceSynchronize();

        //----------------------MATMULT PART----------------------
        end = tagTime();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&event_time_in_millis, begin, stop);

        calc_duration = end - start;
        saveTimeToFile(dim, calc_duration.count(), "csv/calc-cuda-tag.csv");
        saveTimeToFile(dim, event_time_in_millis * 1000000, "csv/calc-cuda-event.csv");

        start = tagTime();
        //----------------------READ PART----------------------
        cudaMemcpy(R, gpu_R, data_size, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        //----------------------READ PART----------------------
        end = tagTime();

        read_duration = end - start;
        saveTimeToFile(dim, read_duration.count(), "csv/read-cuda.csv");
        saveTimeToFile(dim, load_duration.count() + calc_duration.count() + read_duration.count(), "csv/total-cuda.csv");
        saveTimeToFile(dim, load_duration.count() + read_duration.count(), "csv/overhead-cuda.csv");

        if (DEBUG)
        {
            pprintArrayMatrix(R, dim, dim, "naive R");
            printf("Dim: %d, Time: %ldms\n", dim, load_duration.count());
        }

        // Free memory
        free(A);
        free(B);
        free(R);
        cudaFree(gpu_A);
        cudaFree(gpu_B);
        cudaFree(gpu_R);
    }
    cudaEventDestroy(begin);
    cudaEventDestroy(stop);
}