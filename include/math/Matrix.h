#ifndef MATRIX_H
#define MATRIX_H

#include <math/Shape.h>

#include <memory>
#include <vector>

namespace cnncpp {

class Matrix {
 public:
  Shape shape;

  Matrix(size_t x_dim = 0, size_t y_dim = 0);
  Matrix(Shape shape);

  void allocateMemory();
  void allocateMemoryIfNotAllocated(Shape shape);

  float& operator[](const int index);
  const float& operator[](const int index) const;

  float* begin();
  const float* begin() const;

  float* end();
  const float* end() const;

  Matrix operator*(const Matrix& rhs) const;  // CPU matmul

 private:
  bool mIsHostAllocated;
  std::shared_ptr<float> mDataHost;

  void allocateHostMemory();
};

}  // namespace cnncpp

#endif  // MATRIX_H