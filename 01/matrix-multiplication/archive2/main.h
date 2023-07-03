#ifndef MAIN_H_
#define MAIN_H_

#include <iomanip>
#include <iostream>
#include <string>
using namespace std;

void pprintMatrix(float *matrix, int rows, int cols, string name)
{

    int max_width = 0;
    for (int i = 0; i < rows; i += cols)
    {
        for (int j = 0; j < cols; j++)
        {
            int width = to_string(matrix[i + j]).size();
            if (width > max_width)
            {
                max_width = width;
            }
        }
    }

    cout << name << endl;

    // header row

    int left_pad = to_string(rows).size() + 1;

    cout << setw(left_pad) << "  |";
    for (int i = 0; i < cols; ++i)
    {
        cout << " " << setw(max_width) << i << "  ";
    }
    cout << endl;
    cout << string((left_pad + 2), '-')
         << string((max_width + 3) * cols, '-')
         << endl;

    // body

    for (int i = 0; i < rows; i++)
    {
        cout << setw(left_pad) << i << " |";

        for (int j = 0; j < cols; j++)
        {
            cout << " " << setw(max_width) << matrix[i + j] << " |";
        }
        cout << endl;
        cout << string(left_pad + 1, ' ')
             << '-'
             << string((max_width + 3) * cols, '-') << endl;
    }
}

void fillMatrix(float *matrix, int rows, int cols, float value)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            matrix[i + j] = value;
        }
    }
}

#endif