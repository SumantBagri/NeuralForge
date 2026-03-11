#ifndef DATATYPE_H
#define DATATYPE_H

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace nf {

enum class DataType {
    FLOAT32,  // 32-bit floating point
    FLOAT64,  // 64-bit floating point
    FLOAT16,  // 16-bit floating point (half precision)
    INT32,    // 32-bit signed integer
    INT8,     // 8-bit signed integer (quantization)
};

inline size_t getSizeOf(DataType dtype) {
    switch (dtype) {
	case DataType::FLOAT32:
	    return sizeof(float);
	case DataType::FLOAT64:
	    return sizeof(double);
	case DataType::FLOAT16:
	    return 2;  // 16 bits = 2 bytes
	case DataType::INT32:
	    return sizeof(int32_t);
	case DataType::INT8:
	    return sizeof(int8_t);
	default:
	    throw std::invalid_argument("Unknown DataType");
    }
}

inline std::string dataTypeToString(DataType dtype) {
    switch (dtype) {
	case DataType::FLOAT32:
	    return "float32";
	case DataType::FLOAT64:
	    return "float64";
	case DataType::FLOAT16:
	    return "float16";
	case DataType::INT32:
	    return "int32";
	case DataType::INT8:
	    return "int8";
	default:
	    return "unknown";
    }
}

}  // namespace nf

#endif  // DATATYPE_H
