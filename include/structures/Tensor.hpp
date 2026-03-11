#ifndef FLOAT_TENSOR_H
#define FLOAT_TENSOR_H

#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include "structures/DataType.hpp"
#include "structures/Device.hpp"
#include "structures/Shape.hpp"
#include "structures/Storage.hpp"

namespace nf {

// N-dimensional tensor with unified CPU/GPU memory management
// For simplicity, this implementation only supports float32
class FloatTensor {
   public:
    FloatTensor() : FloatTensor(Shape(1)) {}

    explicit FloatTensor(const Shape& shape,
                         const Device& device = Device(DeviceType::CPU))
        : mShape(shape), mDevice(device) {
        // Only CPU storage supported for now
        if (device.isCPU()) {
            mStorage =
                std::make_shared<HostStorage>(shape.numel(), DataType::FLOAT32);
        } else {
            throw std::invalid_argument("CUDA storage not yet implemented");
        }
    }

    FloatTensor(size_t d0, const Device& device = Device(DeviceType::CPU))
        : FloatTensor(Shape(d0), device) {}

    FloatTensor(size_t d0, size_t d1,
                const Device& device = Device(DeviceType::CPU))
        : FloatTensor(Shape(d0, d1), device) {}

    FloatTensor(size_t d0, size_t d1, size_t d2,
                const Device& device = Device(DeviceType::CPU))
        : FloatTensor(Shape(d0, d1, d2), device) {}

    FloatTensor(size_t d0, size_t d1, size_t d2, size_t d3,
                const Device& device = Device(DeviceType::CPU))
        : FloatTensor(Shape(d0, d1, d2, d3), device) {}

    virtual ~FloatTensor() = default;

    const Shape& shape() const { return mShape; }
    DataType dtype() const { return DataType::FLOAT32; }
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

    float& operator[](size_t index) {
        if (index >= numel()) {
            throw std::out_of_range("FloatTensor index out of bounds");
        }
        return dataAs<float>()[index];
    }

    const float& operator[](size_t index) const {
        if (index >= numel()) {
            throw std::out_of_range("FloatTensor index out of bounds");
        }
        return dataAs<float>()[index];
    }

    // --- Memory Management ---
    void zero() { mStorage->zero(); }

    void copyFrom(const FloatTensor& other) {
        if (shape() != other.shape()) {
            throw std::invalid_argument(
                "Shape mismatch in FloatTensor::copyFrom");
        }
        mStorage->copyFrom(*other.mStorage);
    }

    // Size in bytes
    size_t sizeBytes() const { return mStorage->sizeBytes(); }

   protected:
    Shape mShape;
    Device mDevice;
    std::shared_ptr<Storage> mStorage;
};

}  // namespace nf

#endif  // FLOAT_TENSOR_H
