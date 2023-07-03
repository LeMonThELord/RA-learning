#include "helper.h"
#include "tag.h"
#include <iostream>

#define DEBUG (0)

/*For the inversion and the multiplication of the matrices, the execution time of these functions will be measured */
void mat_mul(float **, float **, float **, int, int, int);

int main(int argc, char **argv)
{
    // Setup

    int min_dim, max_dim, step; // args variables

    if (argc != 4)
    {
        printf("Usage: %s <min_dim> <max_dim> <step>\n", argv[0]);
        return 1;
    }

    min_dim = atoi(argv[1]);
    max_dim = atoi(argv[2]);
    step = atoi(argv[3]);

    if (min_dim <= 0 || max_dim <= 0 || step <= 0)
    {
        printf("Error: dimensions must be positive\n");
        return 1;
    }

    // Calc and perf measure
    tag_point start, end;
    tag_segment duration;
    float **A, **B, **R;
    for (int dim = min_dim; dim <= max_dim; dim += step)
    {
        A = createRandomMatrix(dim, dim);
        B = createRandomMatrix(dim, dim);
        R = createZerosMatrix(dim, dim);

        if (DEBUG)
        {
            pprintMatrix(A, dim, dim, "naive A");
            pprintMatrix(B, dim, dim, "naive B");
        }

        start = tagTime();

        //----------------------CRITICAL PART----------------------

        mat_mul(A, B, R, dim, dim, dim);

        //----------------------CRITICAL PART----------------------

        end = tagTime();

        duration = end - start;

        if (DEBUG)
        {
            pprintMatrix(R, dim, dim, "naive R");
            printf("Dim: %d, Time: %ldms\n", dim, duration.count());
            if (!maxtrixMultIsCorrect(A, B, R, dim, dim))
            {
                printf("Multiplied matrix is not correct, aborting...\n");
                return -1;
            }
        }

        // Write the time in the csv file
        saveTimeToFile(dim, duration.count(), "csv/total-naive.csv");

        // Free memory
        free(A);
        free(B);
        free(R);
    }
}

void mat_mul(float **A, float **B, float **R, int R_rows, int M_dims, int R_cols)
{
    for (int i = 0; i < R_rows; i++)
    {
        for (int j = 0; j < R_cols; j++)
        {
            R[i][j] = 0;
            for (int k = 0; k < M_dims; k++)
            {
                R[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return;
}