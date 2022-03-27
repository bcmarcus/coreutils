#ifndef SORT_HELPERS
#define SORT_HELPERS

#include <iostream>
#include <string>
#include <unistd.h>
#include <coreutils/functions/math/simpleMath.hpp>

using namespace coreutils::functions;

namespace coreutils 
{
   namespace functions
   {
      namespace sort {
			template <typename T>
         void swap (T* first, T* second);

         template <typename T>
         void shuffle (T* arr, int size);

         template <typename T>
         void reverse (T* arr, int size);
      }
   }
}

#endif