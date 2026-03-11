#ifndef ABSTRACT_BASE_LAYER_H
#define ABSTRACT_BASE_LAYER_H

#include <string>

#include "structures/Tensor.hpp"

namespace nf {

class AbstractBaseLayer {
   public:
    virtual ~AbstractBaseLayer() = default;

    virtual void forward(const FloatTensor& input) = 0;

    virtual void backward(const FloatTensor& output_gradient) = 0;

    virtual void update(float learningRate) {}

    virtual const FloatTensor& getInput() const = 0;
    virtual const FloatTensor& getOutput() const = 0;
    virtual const FloatTensor& getInputGradient() const = 0;

   protected:
    std::string mName;
};

}  // namespace nf

#endif  // ABSTRACT_BASE_LAYER_H