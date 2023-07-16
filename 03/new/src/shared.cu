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

__global__ void shared_MM(float *A, float *B, float *R, int R_rows, int M_dims, int R_cols)
{
    int const BLOCK_SIZE = 32;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float temp = 0.0f;
    __shared__ float sharedA[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float sharedB[BLOCK_SIZE * BLOCK_SIZE];

    for (int i = 0; i < (M_dims / BLOCK_SIZE); ++i)
    {

        sharedA[row * BLOCK_SIZE + col] = A[row * M_dims + (i * BLOCK_SIZE + col)];
        sharedB[row * BLOCK_SIZE + col] = B[col + (i * BLOCK_SIZE + row) * M_dims];

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            temp += sharedA[row * M_dims + k] * sharedB[k * R_cols + col];
        }

        __syncthreads();
        printf("r4"); // ?
    }

    R[row * R_cols + col] = temp;
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

        // allocate memory to contain the matrices
        cudaMatrixAlloc((void **)&gpu_A, data_size);
        cudaMatrixAlloc((void **)&gpu_B, data_size);
        cudaMatrixAlloc((void **)&gpu_R, data_size);

        //----------------------Arrangement PART----------------------
        int B_cols = dim;
        int A_rows = dim;
        int BLOCK_SIZE = static_cast<int>(sqrt(MaxThreadsPerBlock));
        dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 blocksPerGrid(B_cols / threadsPerBlock.x, A_rows / threadsPerBlock.y);
        //----------------------Arrangement PART----------------------

        start = tagTime();

        //----------------------LOAD PART----------------------

        cudaMemcpy(gpu_A, A, data_size, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_B, B, data_size, cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();

        //----------------------LOAD PART----------------------

        end = tagTime();

        load_duration = end - start;

        // Write the time in the csv file
        saveTimeToFile(dim, load_duration.count(), "csv/load-shared.csv");

        start = tagTime();
        //----------------------MATMULT PART----------------------
        // load and execute the kernel to multiplication into the GPU

        shared_MM<<<blocksPerGrid, threadsPerBlock>>>(gpu_A, gpu_B, gpu_R, dim, dim, dim);
        cudaDeviceSynchronize();

        //----------------------MATMULT PART----------------------
        end = tagTime();

        calc_duration = end - start;
        saveTimeToFile(dim, calc_duration.count(), "csv/calc-shared.csv");

        start = tagTime();
        //----------------------READ PART----------------------
        cudaMemcpy(R, gpu_R, data_size, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        //----------------------READ PART----------------------
        end = tagTime();

        read_duration = end - start;
        saveTimeToFile(dim, read_duration.count(), "csv/read-shared.csv");
        saveTimeToFile(dim, load_duration.count() + calc_duration.count() + read_duration.count(), "csv/total-shared.csv");
        saveTimeToFile(dim, load_duration.count() + read_duration.count(), "csv/overhead-shared.csv");

        if (DEBUG)
        {
            pprintArrayMatrix(R, dim, dim, "naive R");
            printf("Dim: %d, Time: %ldms\n", dim, load_duration.count());
        }

        // Free memory
        delete A;
        delete B;
        delete R;
        cudaFree(gpu_A);
        cudaFree(gpu_B);
        cudaFree(gpu_R);
    }
}