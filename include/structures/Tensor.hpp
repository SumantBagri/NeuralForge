#ifndef TENSOR_H
#define TENSOR_H

#include <cstring>
#include <memory>
#include <stdexcept>
#include <structures/DataType.hpp>
#include <structures/Device.hpp>
#include <structures/Shape.hpp>
#include <structures/Storage.hpp>
#include <vector>

namespace nf {

// N-dimensional tensor with unified CPU/GPU memory management
// Supports multiple precisions and efficient memory access patterns
class Tensor {
   public:
    Tensor() : Tensor(Shape(1)) {}

    explicit Tensor(const Shape& shape, DataType dtype = DataType::FLOAT32,
                    const Device& device = Device(DeviceType::CPU))
        : mShape(shape), mDtype(dtype), mDevice(device) {
	mStrides = computeStrides(shape);

	// Only CPU storage supported for now
	if (device.isCPU()) {
	    mStorage = std::make_shared<HostStorage>(shape.numel(), dtype);
	} else {
	    throw std::invalid_argument("CUDA storage not yet implemented");
	}
    }

    Tensor(size_t d0, DataType dtype = DataType::FLOAT32,
           const Device& device = Device(DeviceType::CPU))
        : Tensor(Shape(d0), dtype, device) {}

    Tensor(size_t d0, size_t d1, DataType dtype = DataType::FLOAT32,
           const Device& device = Device(DeviceType::CPU))
        : Tensor(Shape(d0, d1), dtype, device) {}

    Tensor(size_t d0, size_t d1, size_t d2, DataType dtype = DataType::FLOAT32,
           const Device& device = Device(DeviceType::CPU))
        : Tensor(Shape(d0, d1, d2), dtype, device) {}

    Tensor(size_t d0, size_t d1, size_t d2, size_t d3,
           DataType dtype = DataType::FLOAT32,
           const Device& device = Device(DeviceType::CPU))
        : Tensor(Shape(d0, d1, d2, d3), dtype, device) {}

    virtual ~Tensor() = default;

    const Shape& shape() const { return mShape; }
    const std::vector<size_t>& strides() const { return mStrides; }
    DataType dtype() const { return mDtype; }
    const Device& device() const { return mDevice; }

    size_t numel() const { return mShape.numel(); }
    size_t ndim() const { return mShape.ndim(); }
    size_t size(size_t dim) const { return mShape[dim]; }

    // --- Data Access ---
    void* data() { return mStorage->data(); }
    const void* data() const { return mStorage->data(); }

    template <typename T>
    T* dataAs() {
	return static_cast<T*>(data());
    }

    template <typename T>
    const T* dataAs() const {
	return static_cast<const T*>(data());
    }

    template <typename T>
    T& operator[](size_t index) {
	if (index >= numel()) {
	    throw std::out_of_range("Tensor index out of bounds");
	}
	return dataAs<T>()[index];
    }

    template <typename T>
    const T& operator[](size_t index) const {
	if (index >= numel()) {
	    throw std::out_of_range("Tensor index out of bounds");
	}
	return dataAs<T>()[index];
    }

    float& operator[](size_t index) {
	if (index >= numel()) {
	    throw std::out_of_range("Tensor index out of bounds");
	}
	if (mDtype != DataType::FLOAT32) {
	    throw std::invalid_argument("operator[] requires FLOAT32 dtype");
	}
	return dataAs<float>()[index];
    }

    const float& operator[](size_t index) const {
	if (index >= numel()) {
	    throw std::out_of_range("Tensor index out of bounds");
	}
	if (mDtype != DataType::FLOAT32) {
	    throw std::invalid_argument("operator[] requires FLOAT32 dtype");
	}
	return dataAs<float>()[index];
    }

    // Multi-dimensional indexing
    template <typename T>
    T& at(const std::vector<size_t>& indices) {
	size_t offset = computeOffset(indices);
	return dataAs<T>()[offset];
    }

    template <typename T>
    const T& at(const std::vector<size_t>& indices) const {
	size_t offset = computeOffset(indices);
	return dataAs<const T>()[offset];
    }

    // --- Memory Management ---
    void zero() { mStorage->zero(); }

    void copyFrom(const Tensor& other) {
	if (shape() != other.shape()) {
	    throw std::invalid_argument("Shape mismatch in Tensor::copyFrom");
	}
	if (dtype() != other.dtype()) {
	    throw std::invalid_argument(
	        "DataType mismatch in Tensor::copyFrom");
	}
	mStorage->copyFrom(*other.mStorage);
    }

    // Size in bytes
    size_t sizeBytes() const { return mStorage->sizeBytes(); }

    // --- Storage Access ---
    std::shared_ptr<Storage> storage() { return mStorage; }
    std::shared_ptr<const Storage> storage() const { return mStorage; }

   protected:
    // Compute linear offset from multi-dimensional indices
    size_t computeOffset(const std::vector<size_t>& indices) const {
	if (indices.size() != ndim()) {
	    throw std::invalid_argument("Index dimensionality mismatch: got " +
	                                std::to_string(indices.size()) +
	                                ", expected " + std::to_string(ndim()));
	}

	size_t offset = 0;
	for (size_t i = 0; i < ndim(); ++i) {
	    if (indices[i] >= mShape[i]) {
		throw std::out_of_range("Index out of range in dimension " +
		                        std::to_string(i));
	    }
	    offset += indices[i] * mStrides[i];
	}
	return offset;
    }

    // Compute default row-major strides from shape
    static std::vector<size_t> computeStrides(const Shape& shape) {
	std::vector<size_t> strides;
	size_t stride = 1;
	const auto& dims = shape.dims();

	// Compute strides from rightmost to leftmost dimension
	for (int i = dims.size() - 1; i >= 0; --i) {
	    strides.insert(strides.begin(), stride);
	    stride *= dims[i];
	}
	return strides;
    }

    Shape mShape;
    std::vector<size_t> mStrides;
    DataType mDtype;
    Device mDevice;
    std::shared_ptr<Storage> mStorage;
};

}  // namespace nf

#endif  // TENSOR_H
