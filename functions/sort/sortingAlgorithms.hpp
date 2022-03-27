#ifndef SORTING_ALGORITHMS
#define SORTING_ALGORITHMS

#include <coreutils/functions/sort/sortHelpers.hpp>
#include <coreutils/functions/debug/print.hpp>

namespace coreutils 
{
   namespace functions
   {
      namespace sort {
         template<typename T>
         int* insertionSort (T* arrayPtr, int size);

         template<typename T>
         void merge(T *Arr, int start, int mid, int end);

         // Arr is an array of integer type
         // start and end are the starting and ending index of current interval of Arr
         template<typename T>
         void mergeSort(T *Arr, int start, int end);
		}
   }
}

#endif