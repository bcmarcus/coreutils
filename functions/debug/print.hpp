#ifndef PRINT
#define PRINT

#include <iostream>

namespace coreutils {
   namespace functions {
      namespace debug {
         // prints an array out
         template <typename T>
         int printArr (T* arr, int size, T* exclude = NULL, int excludeSize = 0);

         // dumps all member variables of a function
         template <typename T>
         void var_dump (T value);
		}
	}
}

#endif