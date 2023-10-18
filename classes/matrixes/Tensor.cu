#include <iostream>
#include <random>
#include <cmath>


#include <coreutils/classes/matrixes/Tensor.cuh>

#include <coreutils/functions/debug/print.hpp>
#include <coreutils/functions/sort/sortHelpers.hpp>
#include <coreutils/functions/math/simpleMath.hpp>

#include <coreutils/util/time.hpp>
#include <coreutils/util/cudaErrors.cuh>

using namespace coreutils::functions;
using namespace coreutils::classes::matrixes;

int Tensor::getLength () {
	return this->length;
}

int Tensor::getWidth () {
	return this->width;
}

int Tensor::getHeight () {
	return this->height;
}

float* Tensor::getArr () {
	return arr;
}

long long Tensor::getSize () {
	return memorySize;
}

float* Tensor::getData (int length, int width, int height) {
	if (this->length <= length || this->width <= width || this->height <= height) {
		std::cout << "Invalid input at getData";
		return nullptr;
	}
	return &this->arr[getIndex(length, width, height)];
}

int Tensor::getIndex (int l, int w, int h) const {
	return l * this->width * this->height + w * this->height + h;
}

void Tensor::shuffleEvery () {
	srand(GetTimeStamp().tv_sec + GetTimeStamp().tv_usec);
	for (int length = 0; length < this->length; length++) {
		for (int width = 0; width < this->width; width++) {
			for (int height = 0; height < this->height; height++) {
				sort::swap (&this->arr[getIndex(length, width, height)], 
								&this->arr[getIndex((double) rand() / RAND_MAX * length, (double) rand() / RAND_MAX * width, (double) rand() / RAND_MAX * height)]);

			}
		}
	}
}


int* Tensor::shuffleGroups () {
	int* order = new int[this->length];
	for (int length = 0; length < this->length; length++) {
		srand(GetTimeStamp().tv_sec + GetTimeStamp().tv_usec);
		double randomLength = rand() / RAND_MAX * length;
		order[length] = randomLength;
		for (int width = 0; width < this->width; width++) {
			for (int height = 0; height < this->height; height++) {
				sort::swap (&this->arr [getIndex(length, width, height)], 
								&this->arr [getIndex(randomLength, width, height)]);

			}
		}
	}
	return order;
}

void Tensor::shuffleGroups (int* order) {
	for (int length = 0; length < this->length; length++) {
		for (int width = 0; width < this->width; width++) {
			for (int height = 0; height < this->height; height++) {
				sort::swap (&this->arr [getIndex(length, width, height)], 
								&this->arr [getIndex(order[length], width, height)]);

			}
		}
	}
}

void Tensor::operator += (const Tensor* m2) {
	for (int i = 0; i < length; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < height; k++) {
				this->arr [getIndex(i, j, k)] += m2->arr [getIndex(i, j, k)];
			}
		}
	}
}

void Tensor::operator -= (const Tensor* m2) {
	for (int i = 0; i < length; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < height; k++) {
				this->arr [getIndex(i, j, k)] -= m2->arr [getIndex(i, j, k)];
			}
		}
	}
}

Tensor* Tensor::operator + (const Tensor* m2) {
	Tensor* M3D = new Tensor (this->length, this->width, this->height);

	for (int i = 0; i < length; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < height; k++) {
				M3D->arr[getIndex(i, j, k)] = this->arr [getIndex(i, j, k)] + m2->arr [getIndex(i, j, k)];
			}
		}
	}

	return M3D;
}

Tensor* Tensor::operator - (const Tensor* m2) {
	Tensor* M3D = new Tensor (this->length, this->width, this->height);

	for (int i = 0; i < length; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < height; k++) {
				M3D->arr[getIndex(i, j, k)] = this->arr [getIndex(i, j, k)] - m2->arr [getIndex(i, j, k)];
			}
		}
	}

	return M3D;
}

Tensor* Tensor::operator * (const Tensor* m2) {
	Tensor* M3D = new Tensor(this->length, this->width, this->height);

	for (int i = 0; i < length; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < height; k++) {
				M3D->arr[getIndex(i, j, k)] = this->arr [getIndex(i, j, k)] * m2->arr [getIndex(i, j, k)];
			}
		}
	}

	return M3D;
}

Tensor* Tensor::operator * (const float x) {
	Tensor* M3D = new Tensor (this->length, this->width, this->height);

	for (int i = 0; i < length; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < height; k++) {
				M3D->arr[getIndex(i, j, k)] = this->arr [getIndex(i, j, k)] * x;
			}
		}
	}
	
	return M3D;
}

Tensor* Tensor::operator / (const Tensor* m2) {
	Tensor* M3D = new Tensor (this->length, this->width, this->height);

	for (int i = 0; i < length; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < height; k++) {
				M3D->arr[getIndex(i,j,k)] = this->arr [getIndex(i,j,k)] / m2->arr [getIndex(i,j,k)];
			}
		}
	}
	
	return M3D;
}

bool Tensor::equals (const Tensor* m2, double tolerance) {
	for (int i = 0; i < length; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < height; k++) {
				if (m2->arr[getIndex(i, j, k)] - this->arr [getIndex(i, j, k)] > tolerance) {
					return false;
				}
			}
		}
	}

	return true;
}

void Tensor::randomize (double lowerBound, double upperBound) {
	double currentRandomNumber;
	srand(GetTimeStamp().tv_sec + GetTimeStamp().tv_usec);
	for (int i = 0; i < this->length; i++) {
		for (int j = 0; j < this->width; j++) {
			for (int k = 0; k < this->height; k++) {
				currentRandomNumber = ((double) rand()) / RAND_MAX * (upperBound - lowerBound) + lowerBound;
				this->arr [getIndex(i, j, k)] = currentRandomNumber;
			}
		}
	}
}

void Tensor::xavierRandomize (int l1, int w1, int h1, int l2, int w2, int h2) {
	double bound = sqrt(6) / (sqrt(l1 * w1 * h1 + l2 * w2 * h2));
	this->randomize(-bound, bound);
}

double Tensor::dotProduct (const Tensor* m2) {
	double output = 0;
	
	for (int i = 0; i < length; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < height; k++) {
				output += this->arr [getIndex(i, j, k)] * m2->arr [getIndex(i, j, k)];
			}
		}
	}
	
	return output;
}

double Tensor::sum () {
	double output = 0;
	
	for (int i = 0; i < length; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < height; k++) {
				output += this->arr [getIndex(i, j, k)];
			}
		}
	}
	
	return output;
}

void Tensor::insert (float data, int length, int width, int height) {
	this->arr[getIndex(length, width, height)] = data;
}

void Tensor::printMatrix () const {
	std::cout << '\n' << "{";
	for (int i = 0; i < this->length; i++) {
		std::cout << '\n' << "  {" << '\n';
		for (int j = 0; j < this->width; j++) {
			std::string out = "    {";
			for (int k = 0; k < this->height; k++) {
				out += std::to_string(this->arr [getIndex(i, j, k)]) + ", ";
			}
			out = out.substr(0, out.length () - 2);
			std::cout << out << "}" << '\n';
		}
		std::cout << "  }";
	}
	std::cout << '\n' << "}" << '\n';
}

void Tensor::setMatrix (Tensor* M3D) {
	for (int i = 0; i < length; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < height; k++) {
				this->insert(*M3D->getData(i, j, k), i, j, k);
			}
		}
	}
}

void Tensor::setAll (double x) {
	for (int i = 0; i < length; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < height; k++) {
				this->insert(x, i, j, k);
			}
		}
	}
}

Tensor::Tensor (const int length, const int width, const int height) {
	this->length = length;
	this->width = width;
	this->height = height;
	gpuErrchk(cudaMallocHost((void **) &this->arr, length * width * height * sizeof(float)));
	this->memorySize = length * width * height * sizeof(float);
}

Tensor::Tensor (const Tensor &m3d) {
	this->length = m3d.length;
	this->width = m3d.width;
	this->height = m3d.height;
	gpuErrchk(cudaMallocHost((void **) &this->arr, length * width * height * sizeof(float)));
	gpuErrchk(cudaMemcpy(this->arr, m3d.arr, this->length * this->width * this->height * sizeof(float), cudaMemcpyHostToHost));
	this->memorySize = length * width * height * sizeof(float);
}

Tensor::Tensor () {
	this->length = 0;
	this->width = 0;
	this->height = 0;
	this->arr = nullptr;
	this->memorySize = 0;
}

Tensor::~Tensor () {
	cudaFreeHost(this->arr);
}