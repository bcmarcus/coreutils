#include <iostream>
#include <coreutils/functions/debug/print.hpp>

namespace coreutils {
   namespace functions {
      namespace debug {

         template <typename T>
         int printArr (T* arr, int size, T* exclude, int excludeSize) {
            bool excludeBool = false;
            int count = 0;

            std::cout << "\n\nPrinting array of size: " << size << '\n';

            for (int i = 0; i < size; i++) {
               if (excludeSize != 0) {
                  for (int j = 0; j < excludeSize; j++){
                     if (exclude [j] == arr [i]) {
                        excludeBool = true;
                        break;
                     }
                  }  
               }
               if (excludeBool) {
                  excludeBool = false;
               } else {
                  std::cout << "Array [" << i << "] :: " << arr[i] << '\n';
                  count++;
               }
            }
            std::cout << "There were " << count << " results shown and " << size - count << " omitted" << "\n\n";
            return count;
         }

         template <typename T>
         void var_dump (T value) {
            std::cout << value.toString();
         }
      }
   }
}