#ifndef SIMPLE_MATH
#define SIMPLE_MATH

namespace coreutils 
{
   namespace functions
   {
      namespace math {
         inline bool isPrime (int prime) {
            for (int i = sqrt (prime); i > 1; i--) {
               if (prime % i == 0) {
                  return false;
               }
            }

            return true;
         }

         inline double rand (int beginning, int end) {
            if (beginning > end) {
               std::cout << "Invalid input into rand\n";
               return -1;
            }
				std::srand(time(0));
            return (std::rand() / RAND_MAX * (end - beginning)) + beginning;
         }

         inline bool isPalindrome (int p) {
            for (float i = 6; i > 0; i--){
               if (p % (int) pow (10.0, i) != 0){
                  for (float j = i - 1; j > 0; j--){
                     if (p / (int) pow (10.0, j) % 10 != p % (int) pow (10.0, i - j) / (int) pow (10.0, i - j - 1)){
                        return false;
                     } 
                  }
                  return true;
               }
            }
            return false;
         }

         inline int countDigits (int n)
         {
            int count = 0;
            while (n != 0)
            {
               n = n / 10;
               ++count;
            }
            return count;
         }
		}
	}
}

#endif