#ifndef ABSTRACT_BASE_LAYER_H
#define ABSTRACT_BASE_LAYER_H

#include <math/Matrix.h>

#include <string>

namespace cnncpp {

class AbstractBaseLayer {
 public:
  virtual ~AbstractBaseLayer() = default;

  virtual void forward(const Matrix& input) = 0;

  virtual void backward(const Matrix& output_gradient) = 0;

  virtual void update(float learningRate) {}

  virtual const Matrix& getInput() const = 0;
  virtual const Matrix& getOutput() const = 0;
  virtual const Matrix& getInputGradient() const = 0;

 protected:
  std::string mName;
};

}  // namespace cnncpp

#endif  // ABSTRACT_BASE_LAYER_H