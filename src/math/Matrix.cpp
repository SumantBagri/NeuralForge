#include "math/Matrix.h"

#include <sstream>
#include <stdexcept>

namespace nf {

Matrix::Matrix(size_t x_dim, size_t y_dim)
    : shape(x_dim, y_dim), mIsHostAllocated(false), mDataHost(nullptr) {}

Matrix::Matrix(Shape shape) : Matrix(shape.x, shape.y) {}

float& Matrix::operator[](const int index) { return mDataHost.get()[index]; }

const float& Matrix::operator[](const int index) const {
  return mDataHost.get()[index];
}

float* Matrix::begin() { return mDataHost.get(); }

const float* Matrix::begin() const { return mDataHost.get(); }

float* Matrix::end() { return mDataHost.get() + (shape.x * shape.y); }

const float* Matrix::end() const {
  return mDataHost.get() + (shape.x * shape.y);
}

Matrix Matrix::operator*(const Matrix& rhs) const {
  // check if the shapes are compatible -- this(x1,y1) X other(x2, y2)
  if (this->shape.y != rhs.shape.x) {
    std::ostringstream err;
    err << "Matrix shapes [this(" << this->shape.x << ", " << this->shape.y
        << "), other(" << rhs.shape.x << ", " << rhs.shape.y
        << ")] are not compatible";
    throw std::invalid_argument(err.str());
  }

  Matrix result(this->shape.x, rhs.shape.y);
  // O(n^3)
  for (size_t row = 0; row < this->shape.x; row++) {  // result -> row
    for (size_t col = 0; col < rhs.shape.y; col++) {  // result -> col
      float sum{0.0f};
      for (size_t i = 0; i < this->shape.y; i++) {
	sum += (*this)[row * this->shape.y + i] * rhs[i * rhs.shape.y + col];
      }
      result[row * rhs.shape.y + col] = sum;
    }
  }

  return result;
}

void Matrix::allocateHostMemory() {
  if (!mIsHostAllocated) {
    mDataHost = std::shared_ptr<float>(new float[shape.x * shape.y],
                                       [](float* ptr) { delete[] ptr; });
    mIsHostAllocated = true;
  }
}

void Matrix::allocateMemory() { allocateHostMemory(); }

void Matrix::allocateMemoryIfNotAllocated(Shape shape) {
  if (!mIsHostAllocated) {
    this->shape = shape;
    allocateMemory();
  }
}

}  // namespace nf
