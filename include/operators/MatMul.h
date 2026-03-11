#ifndef MATMUL_H
#define MATMUL_H

#include <structures/Tensor.h>

#include <stdexcept>

namespace nf {

// CPU matrix multiplication utility function for 2D tensors
// Performs C = A @ B (row-major order)
// A: (m, k), B: (k, n) -> Result: (m, n)
inline Tensor matmul(const Tensor& A, const Tensor& B) {
    if (A.ndim() != 2 || B.ndim() != 2) {
	throw std::invalid_argument("matmul: both operands must be 2D tensors");
    }

    if (A.size(1) != B.size(0)) {
	throw std::invalid_argument("matmul: incompatible shapes [(" +
	                            std::to_string(A.size(0)) + ", " +
	                            std::to_string(A.size(1)) + "), (" +
	                            std::to_string(B.size(0)) + ", " +
	                            std::to_string(B.size(1)) + ")]");
    }

    if (A.dtype() != B.dtype()) {
	throw std::invalid_argument("matmul: operands must have same dtype");
    }

    size_t m = A.size(0);
    size_t k = A.size(1);
    size_t n = B.size(1);

    Tensor C(m, n, A.dtype(), A.device());

    const float* a_data = A.dataAs<float>();
    const float* b_data = B.dataAs<float>();
    float* c_data = C.dataAs<float>();

    for (size_t i = 0; i < m; ++i) {
	for (size_t j = 0; j < n; ++j) {
	    float sum = 0.0f;
	    for (size_t t = 0; t < k; ++t) {
		sum += a_data[i * k + t] * b_data[t * n + j];
	    }
	    c_data[i * n + j] = sum;
	}
    }

    return C;
}

}  // namespace nf

#endif  // MATMUL_H
