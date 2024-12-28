#ifndef STATSHPP
#define STATSHPP
#include "vec3.hpp"
#include "hitable.hpp"

__device__ int comparevecs(const void * a, const void * b)
{
    vec3 A = *(vec3*)a;
    vec3 B = *(vec3*)b;
	return (A.length_squared() > B.length_squared()) - A.length_squared() < B.length_squared();
}

__device__ void swapVecs(vec3* arr, int index1, int index2) {
    vec3 color1 = arr[index1];
    vec3 color2 = arr[index2];
    arr[index1] = color2;
    arr[index2] = color1;
}
__device__ void swapHitables(hitable** arr, int index1, int index2) {
    hitable* hitable1 = arr[index1];
    hitable* hitable2 = arr[index2];
    arr[index1] = hitable2;
    arr[index2] = hitable1;
}

__device__ int partitionVecs(vec3* arr, int low, int high, _CoreCrtNonSecureSearchSortCompareFunction cmpfunc) {

    // Choose the pivot
    vec3 pivot = arr[high];

    // Index of smaller element and indicates 
    // the right position of pivot found so far
    int i = low - 1;

    // Traverse arr[low..high] and move all smaller
    // elements on left side. Elements from low to 
    // i are smaller after every iteration
    for (int j = low; j <= high - 1; j++) {
        if (cmpfunc(&arr[j], &pivot)) {
            i++;
            swapVecs(arr, i, j);
        }
    }

    // Move pivot after smaller elements and
    // return its position
    swapVecs(arr, i + 1, high);
    return i + 1;
}

__device__ int partitionHitables(hitable** hitables, int low, int high, _CoreCrtNonSecureSearchSortCompareFunction cmpfunc) {

    // Choose the pivot
    hitable* pivot = hitables[high];

    // Index of smaller element and indicates 
    // the right position of pivot found so far
    int i = low - 1;

    // Traverse arr[low..high] and move all smaller
    // elements on left side. Elements from low to 
    // i are smaller after every iteration
    for (int j = low; j <= high - 1; j++) {
        if (cmpfunc(&hitables[j], &pivot)) {
            i++;
            swapHitables(hitables, i, j);
        }
    }

    // Move pivot after smaller elements and
    // return its position
    swapHitables(hitables, i + 1, high);
    return i + 1;
}

// The QuickSort function implementation
__device__ void quickSortVecs(vec3* colors, int low, int high, _CoreCrtNonSecureSearchSortCompareFunction cmpfunc) {

    if (low < high) {

        // pi is the partition return index of pivot
        int pi = partitionVecs(colors, low, high, cmpfunc);

        // Recursion calls for smaller elements
        // and greater or equals elements
        quickSortVecs(colors, low, pi - 1, cmpfunc);
        quickSortVecs(colors, pi + 1, high, cmpfunc);
    }
}

__device__ void quickSortHitables(hitable** hitables, int low, int high, _CoreCrtNonSecureSearchSortCompareFunction cmpfunc) {

    if (low < high) {

        // pi is the partition return index of pivot
        int pi = partitionHitables(hitables, low, high, cmpfunc);

        // Recursion calls for smaller elements
        // and greater or equals elements
        quickSortHitables(hitables, low, pi - 1, cmpfunc);
        quickSortHitables(hitables, pi + 1, high, cmpfunc);
    }
}

__device__ void sortHitables(hitable** arr, int num, _CoreCrtNonSecureSearchSortCompareFunction cmpfunc) {
    quickSortHitables(arr, 0, num, cmpfunc);
}

__device__ void sortVecs(vec3* colors, int num) {
    quickSortVecs(colors, 0, num, comparevecs);
}
#endif