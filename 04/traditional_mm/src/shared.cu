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

#define gpu_errchk(ans)                        \
    {                                          \
        gpu_assert((ans), __FILE__, __LINE__); \
    }

inline void gpu_assert(cudaError_t code, const char *file, int line,
                       bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "gpu_assert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        exit(code);
    }
}

__global__ void shared_MM(const float *A, const float *B, float *R,
                          int R_rows, int M_dims, int R_cols)
{
    int const TILE_WIDTH = 32;

    __shared__ float sharedA[TILE_WIDTH][TILE_WIDTH]; // Tile size of 32x32
    __shared__ float sharedB[TILE_WIDTH][TILE_WIDTH];

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    sharedA[threadIdx.y][threadIdx.x] = 0.0;
    sharedB[threadIdx.y][threadIdx.x] = 0.0;

    float temp = 0.0;

    for (int ph = 0; ph < (((M_dims - 1) / TILE_WIDTH) + 1); ph++)
    {
        if ((row < R_rows) && (threadIdx.x + (ph * TILE_WIDTH)) < M_dims)
        {
            sharedA[threadIdx.y][threadIdx.x] = A[(row * M_dims) + threadIdx.x + (ph * TILE_WIDTH)];
        }
        if (col < R_cols && (threadIdx.y + ph * TILE_WIDTH) < M_dims)
        {
            sharedB[threadIdx.y][threadIdx.x] = B[(threadIdx.y + ph * TILE_WIDTH) * R_cols + col];
        }
        __syncthreads();

        for (int j = 0; j < TILE_WIDTH; ++j)
        {
            temp += sharedA[threadIdx.y][j] * sharedB[j][threadIdx.x];
        }
    }
    if (row < R_rows && col < R_cols)
    {
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

    for (int dim = min_dim; dim <= max_dim; dim += step)
    {

        if (dim % 10 == 0)
            step = dim / 10;

        float **ones = createOnesMatrix(dim, dim);
        float **zeros = createZerosMatrix(dim, dim);
        A = matrix2ArrayMatrix(ones, dim, dim);
        B = matrix2ArrayMatrix(ones, dim, dim);
        R = matrix2ArrayMatrix(zeros, dim, dim);

        free(ones);
        free(zeros);

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
        dim3 blocksPerGrid((B_cols / threadsPerBlock.x) + 1, (A_rows / threadsPerBlock.y) + 1);
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

        // pprintArrayMatrix(R, dim, dim, "naive R");
        // printf("Dim: %d, Time: %ldms\n", dim, load_duration.count());

        // Free memory
        free(A);
        free(B);
        free(R);
        cudaFree(gpu_A);
        cudaFree(gpu_B);
        cudaFree(gpu_R);
    }
}