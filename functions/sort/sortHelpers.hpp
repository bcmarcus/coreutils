#ifndef SORT_HELPERS
#define SORT_HELPERS

#include <iostream>
#include <string>
#include <unistd.h>

#include <coreutils/functions/math/simpleMath.hpp>

#include <coreutils/util/time.hpp>

using namespace coreutils::functions;

namespace coreutils 
{
   namespace functions
   {
      namespace sort {
			template <typename T>
         inline void swap (T* first, T* second){
            T temp = *first;
            *first = *second;
            *second = temp;
         }

         template <typename T>
         inline void shuffle (T* arr, int size) {
				srand(GetTimeStamp().tv_sec + GetTimeStamp().tv_usec);
            for (int i = 0; i < size; i++) {
					double currentRandomNumber = ((double) rand() / RAND_MAX * size);
               coreutils::functions::sort::swap (&arr[i], &arr[(int) currentRandomNumber]);
            }
         }

         template <typename T>
         inline void reverse (T* arr, int size) {
            for (int i = 0; i < size / 2; i++) {
               coreutils::functions::sort::swap (&arr[i], &arr[size - i - 1]);
            }
         }
      }
   }
}

#endif