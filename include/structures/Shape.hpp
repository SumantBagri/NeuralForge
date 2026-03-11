#ifndef SHAPE_H
#define SHAPE_H

#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace nf {

// N-dimensional shape representation
class Shape {
   public:
    Shape() : mDims({1}) {}
    Shape(size_t d0) : mDims({d0}) {}
    Shape(size_t d0, size_t d1) : mDims({d0, d1}) {}
    Shape(size_t d0, size_t d1, size_t d2) : mDims({d0, d1, d2}) {}
    Shape(size_t d0, size_t d1, size_t d2, size_t d3)
        : mDims({d0, d1, d2, d3}) {}
    Shape(const std::vector<size_t>& dims) : mDims(dims) {
        if (mDims.empty()) {
            throw std::invalid_argument("Shape dimensions cannot be empty");
        }
    }

    size_t ndim() const { return mDims.size(); }

    size_t operator[](size_t i) const {
        if (i >= mDims.size()) {
            throw std::out_of_range("Dimension index out of range");
        }
        return mDims[i];
    }

    size_t numel() const {
        return std::accumulate(mDims.begin(), mDims.end(), size_t(1),
                               std::multiplies<size_t>());
    }

    const std::vector<size_t>& dims() const { return mDims; }

    bool operator==(const Shape& other) const { return mDims == other.mDims; }
    bool operator!=(const Shape& other) const { return !(*this == other); }

   private:
    std::vector<size_t> mDims;
};

}  // namespace nf

#endif  // SHAPE_H
