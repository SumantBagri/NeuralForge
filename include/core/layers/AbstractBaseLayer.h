#ifndef ABSTRACT_BASE_LAYER_H
#define ABSTRACT_BASE_LAYER_H

#include <structures/Tensor.h>

#include <string>

namespace nf {

class AbstractBaseLayer {
 public:
  virtual ~AbstractBaseLayer() = default;

  virtual void forward(const Tensor& input) = 0;

  virtual void backward(const Tensor& output_gradient) = 0;

  virtual void update(float learningRate) {}

  virtual const Tensor& getInput() const = 0;
  virtual const Tensor& getOutput() const = 0;
  virtual const Tensor& getInputGradient() const = 0;

 protected:
  std::string mName;
};

}  // namespace nf

#endif  // ABSTRACT_BASE_LAYER_H