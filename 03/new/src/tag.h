#ifndef TAG_H_
#define TAG_H_

#include <chrono>

typedef chrono::high_resolution_clock::time_point tag_point;
typedef chrono::high_resolution_clock::duration tag_segment;

void tagPrint(tag_point t1, tag_point t2)
{
    cout << "Time: " << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() << "ms" << endl;
}

void tagHighRes(tag_point t1, tag_point t2)
{
    cout << "Time: " << chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count() << "ns" << endl;
}

tag_point tagTime()
{
    return chrono::high_resolution_clock::now();
}

long tagMillis(tag_point t1)
{
    tag_point t2 = chrono::high_resolution_clock::now();
    return chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
}

long tagNanos(tag_point t1)
{
    tag_point t2 = chrono::high_resolution_clock::now();
    return chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count();
}

tag_point tagTillNow(tag_point t1)
{
    tag_point t2 = chrono::high_resolution_clock::now();
    return t2;
}

#endif