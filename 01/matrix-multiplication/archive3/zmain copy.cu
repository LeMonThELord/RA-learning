#include "matrix.cpp"
#include "tag.cpp"
#include <iostream>

using namespace std;

#define N_MIN 10
#define N_MAX 10000

void MM_check(Matrix &left, Matrix &right, Matrix &result)
{
    if (left.getCols() != right.getRows())
    {
        cout << "Error: Matrix dimensions do not match." << endl;
        exit(1);
    }

    if (right.getCols() != result.getCols() || left.getRows() != result.getRows())
    {
        cout << "Error: Result matrix has wrong dimensions." << endl;
        exit(1);
    }
}

typedef struct
{
    int rows;
    int cols;
    float *data;
} device_matrix_t;

__global__ void matrixMul(const device_matrix_t left, const device_matrix_t right, device_matrix_t result)
{
    // Compute each thread's global row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float temp_val = 0;

    // Iterate over row, and down column
    for (int k = 0; k < left.getCols(); k++)
    {
        // Accumulate results for a single element
        temp_val += left.data[row * left.cols + k] * right.data[k * right.cols + col];
    }

    result.data[row * result.cols + col] = temp_val;
}

void cuda_MM(Matrix &left, Matrix &right, Matrix &result)
{
    MM_check(left, right, result);

    int left_rows = left.getRows();
    int left_cols = left.getCols();
    int right_rows = right.getRows();
    int right_cols = right.getCols();
    int result_rows = result.getRows();
    int result_cols = result.getCols();

    // Allocate memory on the device
    device_matrix_t d_left;
    d_left.rows = left_rows;
    d_left.cols = left_cols;
    int left_size = left_rows * left_cols * sizeof(float);
    d_left.data = cudaMalloc(&d_left.data, left_size);
    for (int i = 0; i < left_rows; i++)
    {
        for (int j = 0; j < left_cols; j++)
        {
            d_left.data[i * left_cols + j] = left.getMatrix()[i][j];
        }
    }
    device_matrix_t d_right;
    d_right.rows = right_rows;
    d_right.cols = right_cols;
    int right_size = right_rows * right_cols * sizeof(float);
    d_right.data = cudaMalloc(&d_right.data, right_size);
    for (int i = 0; i < right_rows; i++)
    {
        for (int j = 0; j < right_cols; j++)
        {
            d_right.data[i * right_cols + j] = right.getMatrix()[i][j];
        }
    }
    device_matrix_t d_result;
    d_result.rows = result_rows;
    d_result.cols = result_cols;
    int result_size = result_rows * result_cols * sizeof(float);
    int result_size = result_rows * result_cols * sizeof(float);
    d_result.data = cudaMalloc(&d_result.data, result_size);
    for (int i = 0; i < result_rows; i++)
    {
        for (int j = 0; j < result_cols; j++)
        {
            d_result.data[i * result_cols + j] = 0;
        }
    }

    // Launch the kernel
    int BLOCK_SIZE = 32;
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid(result_cols / threadsPerBlock.x, result_rows / threadsPerBlock.y);
    if (N * N > 1024)
    {
        threadsPerBlock.x = 1024;
        threadsPerBlock.y = 1024;
        blocksPerGrid.x = ceil(double(N) / double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(N) / double(threadsPerBlock.y));
    }
    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_left, d_right, d_result);

    // Copy back the result
    cudaMemcpy(result.getData(), d_result, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_left);
    cudaFree(d_right);
    cudaFree(d_result);
}

int main()
{
    int N = 100;

    if (N < N_MIN || N > N_MAX)
    {
        cout << "N must be between " << N_MIN << " and " << N_MAX << endl;
        return 1;
    }

    time_point start_tag;

    Matrix m1(N, N);
    m1.fill(1);

    Matrix m2(N, N);
    m2.fill(1);

    Matrix mr(N, N);

    MM_check(m1, m2, mr);

    start_tag = tagTime();
    cuda_MM(m1, m2, mr);
    tagPrint(start_tag, tagTillNow(start_tag));
    // mr.pprintMatrix("mr");
}