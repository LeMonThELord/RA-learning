#include "helper.h"
#include "tag.h"
#include <iostream>

#define DEBUG (0)

#define MIN(a, b) (((a) > (b)) ? (b) : (a))

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

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < R_rows; i += stride)
    {
        float *temp = new float[R_cols];
        for (int j = 0; j < R_cols; j++)
        {
            temp[j] = 0;
        }
        for (int j = 0; j < R_cols; j++)
        {
            for (int k = 0; k < M_dims; k++)
            {
                temp[k] += A[i * M_dims + j] * B[j * R_cols + k];
            }
        }
        memcpy(&R[i * R_cols], temp, R_cols * sizeof(float));
        delete[] temp;
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

    int threads_count = 1;
    if (argc == 5)
        threads_count = atoi(argv[4]);

    // Hardware info
    cudaDeviceProp gpu_prop;
    cudaGetDeviceProperties(&gpu_prop, 0);
    int MaxThreadsPerBlock = gpu_prop.maxThreadsPerBlock;

    // Calc and perf measure
    float *A, *B, *R;
    float *gpu_A, *gpu_B, *gpu_R;
    tag_point start, end;
    tag_segment load_duration, calc_duration, read_duration;

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

        int threadsPerBlock = MIN(threads_count, MaxThreadsPerBlock);
        int blocksPerGrid = 1;

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

        if (DEBUG)
        {
            pprintArrayMatrix(R, dim, dim, "naive R");
            printf("Dim: %d, Time: %ldms\n", dim, load_duration.count());
        }

        // Write the time in the csv file
        saveTimeToFile(dim, load_duration.count(), "csv/load-row-cuda-" + to_string(threads_count) + ".csv");

        start = tagTime();
        //----------------------MATMULT PART----------------------
        // load and execute the kernel to multiplication into the GPU

        matrixMultiplication<<<blocksPerGrid, threadsPerBlock>>>(gpu_A, gpu_B, gpu_R, dim, dim, dim);
        cudaDeviceSynchronize();

        //----------------------MATMULT PART----------------------
        end = tagTime();

        calc_duration = end - start;
        saveTimeToFile(dim, calc_duration.count(), "csv/calc-row-cuda-" + to_string(threads_count) + ".csv");

        start = tagTime();
        //----------------------READ PART----------------------
        cudaMemcpy(R, gpu_R, data_size, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        //----------------------READ PART----------------------
        end = tagTime();

        read_duration = end - start;
        saveTimeToFile(dim, read_duration.count(), "csv/read-row-cuda-" + to_string(threads_count) + ".csv");
        saveTimeToFile(dim, load_duration.count() + calc_duration.count() + read_duration.count(), "csv/total-row-cuda-" + to_string(threads_count) + ".csv");
        saveTimeToFile(dim, load_duration.count() + read_duration.count(), "csv/overhead-row-cuda-" + to_string(threads_count) + ".csv");

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
}