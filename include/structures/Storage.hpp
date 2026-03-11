#ifndef STORAGE_H
#define STORAGE_H

#include <cstring>
#include <stdexcept>

#include "structures/DataType.hpp"
#include "structures/Device.hpp"

namespace nf {

// Abstract base class for tensor data storage (host or device)
class Storage {
   public:
    virtual ~Storage() = default;

    virtual void* data() = 0;
    virtual const void* data() const = 0;

    virtual size_t numel() const = 0;

    virtual const Device& device() const = 0;

    virtual DataType dtype() const = 0;

    virtual size_t sizeBytes() const = 0;

    // Copy from another storage (handles CPU/GPU transfers)
    virtual void copyFrom(const Storage& other) = 0;

    // Fill storage with zeros
    virtual void zero() = 0;
};

// Host (CPU) memory storage with GPU-aligned allocation
class HostStorage : public Storage {
   public:
    HostStorage(size_t numel, DataType dtype)
        : mNumel(numel), mDtype(dtype), mDevice(DeviceType::CPU, 0) {
        allocateAligned();
    }

    ~HostStorage() {
        if (mData) {
            free(mData);
        }
    }

    void* data() override { return mData; }
    const void* data() const override { return mData; }

    size_t numel() const override { return mNumel; }
    const Device& device() const override { return mDevice; }
    DataType dtype() const override { return mDtype; }
    size_t sizeBytes() const override { return mNumel * getSizeOf(mDtype); }

    void copyFrom(const Storage& other) override {
        if (other.numel() != mNumel) {
            throw std::invalid_argument("Storage size mismatch in copyFrom: " +
                                        std::to_string(other.numel()) + " vs " +
                                        std::to_string(mNumel));
        }
        if (other.dtype() != mDtype) {
            throw std::invalid_argument("DataType mismatch in copyFrom");
        }
        if (other.device().isCPU()) {
            std::memcpy(mData, other.data(), sizeBytes());
        } else {
            throw std::invalid_argument(
                "copyFrom: GPU to CPU copy not implemented yet");
        }
    }

    void zero() override { std::memset(mData, 0, sizeBytes()); }

   private:
    void allocateAligned() {
        // Align to 256 bytes for GPU memory coalescing efficiency
        const size_t alignment = 256;
        size_t bytes = sizeBytes();

        // Use posix_memalign for cache-line and GPU alignment
        int ret = posix_memalign(&mData, alignment, bytes);
        if (ret != 0) {
            throw std::bad_alloc();
        }
    }

    void* mData = nullptr;
    size_t mNumel;
    DataType mDtype;
    Device mDevice;
};

}  // namespace nf

#endif  // STORAGE_H
