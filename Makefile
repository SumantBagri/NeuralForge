.PHONY: build format test clean help

help:
	@echo "Available targets:"
	@echo "  make build    - Build the workspace"
	@echo "  make format   - Format C++ and Python files"
	@echo "  make test     - Run unit tests"
	@echo "  make clean    - Remove build artifacts"

build:
	@echo "Building workspace..."
	cmake -G Ninja -S . -B build
	cmake --build build

format:
	@echo "Formatting C++ files..."
	@for file in $$(find include src -name "*.hpp" -o -name "*.cpp"); do \
		echo "  Formatting $$file"; \
		clang-format -i "$$file"; \
	done
	@echo "Formatting Python files..."
	python3 -m black . 2>/dev/null || true

test: build
	@echo "Running unit tests..."
	cd build && ctest --output-on-failure

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build
	find . -name "*.o" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
