#ifndef Tensor_CUH
#define Tensor_CUH

namespace coreutils {
   namespace classes {
      namespace matrixes {
         class Tensor{
				private:
               int length;
               int width;
               int height;
               long long memorySize;
               float* arr;

            public:

               int getLength ();

               int getWidth ();

               int getHeight ();

               float* getArr ();

               long long getSize ();

               float* getData (int length, int width, int height);

               int getIndex (int l, int w, int h) const;

               // shuffles every single value
               void shuffleEvery ();

               // shuffles every 2d matrix. while retaining the 2d matrix
               int* shuffleGroups ();
               void shuffleGroups (int* order);

               // adds this and another matrix and 
               // sets this matrix equal to it
               void operator += (const Tensor* m2);

               // subtracts this and another matrix and 
               // sets this matrix equal to it 
               void operator -= (const Tensor* m2);

               // returns addition of this and another matrix
               Tensor* operator + (const Tensor* m2);

               // returns subtraction of this and another matrix
               Tensor* operator - (const Tensor* m2);

               // returns multiplication of this and another matrix
               Tensor* operator * (const Tensor* m2);
					
					// returns this matrix scaled by a value
               Tensor* operator * (const float x);

					// returns division of this and another matrix
               Tensor* operator / (const Tensor* m2);

					bool equals (const Tensor* m2, double tolerance = 0.001);

               void randomize (double lowerBound = -0.5, double upperBound = 0.5);

					void xavierRandomize (int l1, int w1, int h1, int l2, int w2, int h2);

               double dotProduct (const Tensor* m2);

               double sum ();

               void insert (float data, int length, int width, int height);

               void printMatrix () const;

               void setMatrix (Tensor* M3D);

					void setAll (double x);

               Tensor (const int length, const int width, const int height);
					Tensor (const Tensor& m3d);
               Tensor ();

               ~Tensor ();
         };
      }
   }
}

#endif