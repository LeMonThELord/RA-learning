#include <chrono>

typedef chrono::high_resolution_clock::time_point time_point;

void tagPrint(time_point t1, time_point t2)
{
    cout << "Time: " << chrono::duration_cast<chrono::microseconds>(t2 - t1).count() << "µs" << endl;
}

time_point tagTime()
{
    return chrono::high_resolution_clock::now();
}

long tagMillis(time_point t1)
{
    time_point t2 = chrono::high_resolution_clock::now();
    return chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
}

long tagNanos(time_point t1)
{
    time_point t2 = chrono::high_resolution_clock::now();
    return chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count();
}

time_point tagTillNow(time_point t1)
{
    time_point t2 = chrono::high_resolution_clock::now();
    return t2;
}