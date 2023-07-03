#include "helper.h"
#include "tag.h"
#include <iostream>

#define DEBUG (0)

#define MIN(a, b) (((a) > (b)) ? (b) : (a))

void *thread_mat_mul(void *);

typedef struct
{
    float **A;
    float **B;
    float **R;
    int R_rows;
    int M_dims;
    int R_cols;
    int from_row;
    int to_row;
} param_t;

int main(int argc, char **argv)
{

    // Setup

    int min_dim, max_dim, step; // args variables

    if (argc < 4 || argc > 5)
    {
        printf("Usage: %s <min_dim> <max_dim> <step> [<thread_num>]\n", argv[0]);
        return 1;
    }

    min_dim = atoi(argv[1]);
    max_dim = atoi(argv[2]);
    step = atoi(argv[3]);

    int threads_count = 1;
    if (argc == 5)
        threads_count = atoi(argv[4]);

    static pthread_barrier_t barrier;
    pthread_t *threads = new pthread_t[threads_count];
    param_t *params = new param_t[threads_count];

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
            pprintMatrix(A, dim, dim, "pthread A");
            pprintMatrix(B, dim, dim, "pthread B");
        }

        pthread_barrier_init(&barrier, NULL, threads_count);

        start = tagTime();

        //----------------------CRITICAL PART----------------------

        for (int i = 0; i < threads_count; i++)
        {
            params[i].A = A;
            params[i].B = B;
            params[i].R = R;
            params[i].R_rows = dim;
            params[i].M_dims = dim;
            params[i].R_cols = dim;
            params[i].from_row = i * dim / threads_count;
            params[i].to_row = MIN((i + 1) * dim / threads_count, dim);
            pthread_create(&threads[i], NULL, thread_mat_mul, &params[i]);
        }

        for (int i = 0; i < threads_count; i++)
        {
            pthread_join(threads[i], NULL);
        }

        //----------------------CRITICAL PART----------------------

        end = tagTime();

        duration = end - start;

        if (DEBUG)
        {
            pprintMatrix(R, dim, dim, "naive R");
            printf("Dim: %d, Time: %ldms\n", dim, duration.count());
        }

        pthread_barrier_destroy(&barrier);

        // Write the time in the csv file
        saveTimeToFile(dim, duration.count(), "csv/total-pthread-" + to_string(threads_count) + ".csv");

        // Free memory
        free(A);
        free(B);
        free(R);
    }
}
void *thread_mat_mul(void *params)
{
    param_t *p = (param_t *)params;
    int R_rows = p->R_rows;
    int M_dims = p->M_dims;
    int R_cols = p->R_cols;
    float **A = p->A;
    float **B = p->B;
    float **R = p->R;
    int from_row = p->from_row;
    int to_row = p->to_row;

    int i, j, k;
    float temp;

    for (i = from_row; i < to_row; i++)
    {
        for (j = 0; j < R_cols; j++)
        {
            temp = 0;
            for (k = 0; k < M_dims; k++)
            {
                temp += A[i][k] * B[k][j];
            }
            R[i][j] = temp;
        }
    }

    return NULL;
}