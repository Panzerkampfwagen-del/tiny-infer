CXX      ?= c++
CFLAGS   := -std=c++17 -O2 -g -Wall -Wextra -Iinclude -fopenmp
BUILD    := build
DEPS     := include/tensor.hpp

.PHONY: all test bench clean

all: $(BUILD)/test_infer $(BUILD)/bench_matmul

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/test_infer: tests/test_infer.cpp src/tensor.cpp $(DEPS) | $(BUILD)
	$(CXX) $(CFLAGS) $^ -o $@

$(BUILD)/bench_matmul: bench/bench_matmul.cpp src/tensor.cpp $(DEPS) | $(BUILD)
	$(CXX) $(CFLAGS) $^ -o $@

test: $(BUILD)/test_infer
	./$(BUILD)/test_infer

bench: $(BUILD)/bench_matmul
	./$(BUILD)/bench_matmul

clean:
	rm -rf $(BUILD)
