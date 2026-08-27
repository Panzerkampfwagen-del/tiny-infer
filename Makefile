CXX      ?= c++
CFLAGS   := -std=c++17 -O2 -g -Wall -Wextra -Iinclude -fopenmp
BUILD    := build
DEPS     := include/tensor.hpp include/model.hpp include/generate.hpp \
            include/tokenizer.hpp
COMMON   := src/tensor.cpp src/model.cpp src/generate.cpp src/tokenizer.cpp

.PHONY: all test test-model bench bench-batch clean

all: $(BUILD)/test_infer $(BUILD)/test_model $(BUILD)/bench_matmul \
     $(BUILD)/bench_batch $(BUILD)/tinyinfer_cli

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/test_infer: tests/test_infer.cpp src/tensor.cpp $(DEPS) | $(BUILD)
	$(CXX) $(CFLAGS) $^ -o $@

$(BUILD)/test_model: tests/test_model.cpp $(COMMON) $(DEPS)
	@mkdir -p $(BUILD)
	$(CXX) $(CFLAGS) $^ -o $@

$(BUILD)/bench_matmul: bench/bench_matmul.cpp src/tensor.cpp $(DEPS) | $(BUILD)
	$(CXX) $(CFLAGS) $^ -o $@

$(BUILD)/bench_batch: bench/bench_batch.cpp $(COMMON) $(DEPS)
	@mkdir -p $(BUILD)
	$(CXX) $(CFLAGS) $^ -o $@

$(BUILD)/tinyinfer_cli: src/cli_main.cpp $(COMMON) $(DEPS)
	@mkdir -p $(BUILD)
	$(CXX) $(CFLAGS) $^ -o $@

test: $(BUILD)/test_infer
	./$(BUILD)/test_infer

test-model: $(BUILD)/test_model
	./$(BUILD)/test_model

bench: $(BUILD)/bench_matmul
	./$(BUILD)/bench_matmul

bench-batch: $(BUILD)/bench_batch
	./$(BUILD)/bench_batch

clean:
	rm -rf $(BUILD)
