#include <iostream>
#include <string>
#include <random>
#include <coreutils/util/time.hpp>
#include <unistd.h>
#include <coreutils/functions/sort/sortHelpers.hpp>
#include <coreutils/functions/math/simpleMath.hpp>

using namespace coreutils::functions;

namespace coreutils 
{
   namespace functions
   {
      namespace sort {
         template <typename T>
         void swap (T* first, T* second){
            T temp = *first;
            *first = *second;
            *second = temp;
         }

         template <typename T>
         void shuffle (T* arr, int size) {
				srand(GetTimeStamp().tv_sec + GetTimeStamp().tv_usec);
            for (int i = 0; i < size; i++) {
					double currentRandomNumber = ((double) rand() / RAND_MAX * size);
               coreutils::functions::sort::swap (&arr[i], &arr[(int) currentRandomNumber]);
            }
         }

         template <typename T>
         void reverse (T* arr, int size) {
            for (int i = 0; i < size / 2; i++) {
               coreutils::functions::sort::swap (&arr[i], &arr[size - i - 1]);
            }
         }
      }
   }
}