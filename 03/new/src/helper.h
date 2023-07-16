#ifndef HELPER_H_
#define HELPER_H_

#include <fstream>
#include <iomanip>
#include <iostream>

using namespace std;

#define MY_RAND_MAX (10)
#define ERROR_THRESHOLD (0.0001)

float **createMatrixWithGenerator(int rows, int cols, float (*generator)())
{
    float **result;
    result = new float *[rows];

    for (int r = 0; r < rows; r++)
    {
        result[r] = new float[cols];

        for (int w = 0; w < cols; w++)
            result[r][w] = generator();
    }

    return result;
}

float **createOnesMatrix(int rows, int cols)
{
    return createMatrixWithGenerator(rows, cols, []()
                                     { return 1.0f; });
}

float **createZerosMatrix(int rows, int cols)
{
    return createMatrixWithGenerator(rows, cols, []()
                                     { return 0.0f; });
}

float **createRandomMatrix(int rows, int cols, int seed = time(NULL))
{
    srand(seed);
    return createMatrixWithGenerator(rows, cols, []()
                                     { return (float)rand() / MY_RAND_MAX; });
}

float *matrix2ArrayMatrix(float **matrix, int rows, int cols)
{
    float *result = new float[rows * cols];

    for (int r = 0, i = 0; r < rows; r++)
    {
        for (int w = 0; w < cols; w++, i++)
            result[i] = matrix[r][w];

        delete matrix[r];
    }

    delete matrix;

    return result;
}

float **arrayMatrix2Matrix(float *arrayMatrix, int rows, int cols)
{
    float **result = new float *[rows];

    for (int r = 0, i = 0; r < rows; r++)
    {
        result[r] = new float[cols];

        for (int w = 0; w < cols; w++, i++)
            result[r][w] = arrayMatrix[i];
    }

    delete arrayMatrix;

    return result;
}

void pprintMatrix(float **matrix, int rows, int cols, string name)
{
    // get max width of elements
    int max_width = 0;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            max_width = std::max(max_width, static_cast<int>(std::to_string(matrix[i][j]).size()));
        }
    }

    // print title
    printf("\nMATRIX: %s\n\n", name.c_str());

    // print header row

    int left_pad = std::to_string(rows).size() + 1;

    cout << setw(left_pad) << "    |";

    for (size_t i = 0; i < cols; ++i)
    {
        cout << " " << setw(max_width) << i << "  ";
    }
    printf("\n%s\n", string((left_pad) + ((max_width + 3) * cols), '-').c_str());

    // body

    for (int i = 0; i < rows; i++)
    {
        cout << setw(left_pad) << i << " |";

        for (int j = 0; j < cols; j++)
        {
            cout << " " << setw(max_width) << matrix[i][j] << " |";
        }
        printf("\n%s%s\n",
               string(left_pad + 1, ' ').c_str(),
               string((1) + ((max_width + 3) * cols), '-').c_str());
    }
}

void pprintArrayMatrix(float *arrayMatrix, int rows, int cols, string name)
{
    pprintMatrix(arrayMatrix2Matrix(arrayMatrix, rows, cols), rows, cols, name);
}

bool maxtrixMultIsCorrect(float **A, float **B, float **R, int rows, int cols)
{
    bool *error = new bool[rows * cols];
    float **R2 = createZerosMatrix(rows, cols);

    for (int r = 0, i = 0; r < rows; r++)
    {
        for (int w = 0; w < cols; w++, i++)
        {
            R2[r][w] = 0;
            for (int k = 0; k < cols; k++)
            {
                R2[r][w] += A[r][k] * B[k][w];
            }
            error[i] = (R2[r][w] != R[r][w]);
        }
    }

    int errors = 0;
    for (int r = 0, i = 0; r < rows; r++)
    {
        for (int w = 0; w < cols; w++, i++)
        {
            if (error[i])
            {
                errors++;
                printf("[Error] R[%d][%d] | Expected: %f | Got: %f\n", r, w, R2[r][w], R[r][w]);
            }
        }
    }

    float error_rate = (float)errors / (rows * cols);
    printf("Error rate: %f\n", error_rate);
    float error_threshold = ERROR_THRESHOLD;
    if (error_rate > error_threshold)
    {
        printf("Error rate is above threshold (%f > %f)\n", error_rate, error_threshold);
    }

    free(error);
    return true;
}

// From Tecnarca/CPU-GPU-speed-comparison
/*The following function saves the dim (x) and the recorded time (y)
  to a file (*filename) in the format: x y */
void saveTimeToFile(long x, double y, string filename)
{
    ofstream file;
    file.open(filename, ios_base::app);
    file << x << " " << fixed << y / 1000000000 << scientific << endl;
    printf("Saved dim: %ld time: %f to %s\n", x, y / 1000000000, filename.c_str());
    file.close();
}

#endif