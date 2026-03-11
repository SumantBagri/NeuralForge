# ⚠️ NeuralForge - Work In Progress

> **Note**: This project is under active development for **research and learning purposes**. It is **not recommended for production systems**.

A lightweight C++23 neural network framework designed for deep learning research and education. Inspired by tiny-cuda-nn and TinyNeRF.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-23-blue.svg)](https://en.cppreference.com/w/cpp/23)

## Overview

NeuralForge provides a clean API for building and training neural networks in C++. It emphasizes **simplicity**, **readability**, and **educational value**.

### Key Characteristics

- **Modern C++23**: Type-safe, efficient implementation
- **Clean Architecture**: Well-defined layer abstractions for easy extension
- **Matrix-Centric Design**: Efficient tensor operations at the core
- **Learning-Focused**: Designed for research and educational use

## Features

**Layers**: Conv2d, Fully Connected, MaxPool2d, ReLU, Sigmoid  
**Training**: Cross-Entropy Loss, SGD Optimizer, Forward/Backward Propagation  
**Utilities**: Matrix operations, Data loading, Shape inference

## Quick Build

```bash
git clone git@github.com:SumantBagri/NeuralForge.git
cd NeuralForge

# Build (requires Clang 15+, CMake 3.10+)
make build

# Run tests
make test

# Format code
make format
```

## Contributing

Contributions are welcome for research and learning!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Follow code style: `make format`
4. Add tests for new functionality
5. Submit a pull request

## Roadmap

- [ ] CUDA/GPU acceleration support
- [ ] Additional layer types (BatchNorm, Dropout, etc.)
- [ ] More optimizers (Adam, RMSprop)
- [ ] Quantization support
- [ ] Distributed training frameworks
- [ ] Pre-trained model zoo
- [ ] Python bindings via pybind11

## License

MIT License - See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{neuralforge2026,
  title={NeuralForge: A C++ Neural Network Framework for Research},
  author={Bagri, Sumant},
  year={2026},
  url={https://github.com/SumantBagri/NeuralForge}
}
```

## Acknowledgments

Inspired by:
- [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)
- [TinyNeRF](https://github.com/bmild/nerf)

---

**Questions?** Open an issue on [GitHub Issues](https://github.com/SumantBagri/NeuralForge/issues)
