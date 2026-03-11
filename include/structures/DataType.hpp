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
    INT8,     // 8-bit signed integer (quantization)
};

template <DataType D>
struct DataTypeTraits;

template <>
struct DataTypeTraits<DataType::FLOAT32> {
    using type = float;
};

template <>
struct DataTypeTraits<DataType::FLOAT64> {
    using type = double;
};

template <>
struct DataTypeTraits<DataType::INT8> {
    using type = int8_t;
};

template <DataType D>
using dtype_t = typename DataTypeTraits<D>::type;

inline size_t getSizeOf(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32:
            return sizeof(dtype_t<DataType::FLOAT32>);
        case DataType::FLOAT64:
            return sizeof(dtype_t<DataType::FLOAT64>);
        case DataType::INT8:
            return sizeof(dtype_t<DataType::INT8>);
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
        case DataType::INT8:
            return "int8";
        default:
            return "unknown";
    }
}

}  // namespace nf

#endif  // DATATYPE_H
