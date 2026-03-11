#ifndef DEVICE_H
#define DEVICE_H

#include <string>

namespace nf {

enum class DeviceType {
    CPU,   // CPU host memory
    CUDA,  // NVIDIA GPU (with CUDA support)
};

struct Device {
    DeviceType type;
    int index;  // GPU index for multi-GPU systems

    Device(DeviceType type = DeviceType::CPU, int index = 0)
        : type(type), index(index) {}

    bool isCPU() const { return type == DeviceType::CPU; }
    bool isCUDA() const { return type == DeviceType::CUDA; }

    std::string toString() const {
	if (isCPU())
	    return "CPU";
	return "CUDA:" + std::to_string(index);
    }

    bool operator==(const Device& other) const {
	return type == other.type && index == other.index;
    }
};

}  // namespace nf

#endif  // DEVICE_H
