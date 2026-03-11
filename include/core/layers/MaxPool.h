#ifndef MAXPOOL_H
#define MAXPOOL_H

#include <core/layers/AbstractBaseLayer.h>

#include <vector>

namespace nf {

class MaxPool : public AbstractBaseLayer {
 public:
  MaxPool(size_t poolSize = 2, size_t stride = 2,
          std::string name = "MaxPool2D");
  ~MaxPool() override = default;

  void forward(const Matrix& input) override;
  void backward(const Matrix& output_gradient) override;

  const Matrix& getInput() const override { return mInput; }
  const Matrix& getOutput() const override { return mOutput; }
  const Matrix& getInputGradient() const override { return mInputGradient; }

 private:
  size_t mPoolSize;
  size_t mStride;
  Matrix mInput;
  Matrix mOutput;
  Matrix mInputGradient;
  std::vector<size_t> mMaxIndices;
};

}  // namespace nf

#endif  // MAXPOOL_H