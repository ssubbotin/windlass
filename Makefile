# windlass — MoE inference with NVMe-streamed experts
# Copyright (c) 2026 Sergey Subbotin <ssubbotin@gmail.com>
# SPDX-License-Identifier: MIT

CUDA_HOME ?= $(firstword $(wildcard /usr/local/cuda-13 /usr/local/cuda-12.8 /usr/local/cuda))
NVCC       = $(CUDA_HOME)/bin/nvcc
CUDA_LIB   = $(CUDA_HOME)/targets/x86_64-linux/lib

# Blackwell (RTX PRO 6000 / RTX 50xx) = sm_120. Ada = sm_89. Hopper = sm_90.
ARCH     ?= sm_120
CFLAGS    = -O2 -Wno-deprecated-gpu-targets -arch=$(ARCH)
LDFLAGS   = -lpthread

SRC      = src
HEADERS  = $(SRC)/glm_primitives.cuh $(SRC)/glm_kernels.cuh $(SRC)/glm_loader.cuh \
           $(SRC)/glm_layer_runner.cuh $(SRC)/glm_expert_cache.cuh $(SRC)/safetensors_io.cuh

BINARIES = infer_glm test_glm_chain test_glm_layer test_glm_expert_cache test_glm_mxfp4 \
           test_glm_indexer_loader test_glm_indexer

all: infer_glm

infer_glm: $(SRC)/infer_glm.cu $(HEADERS)
	$(NVCC) $(CFLAGS) -I$(SRC) -o $@ $< $(LDFLAGS)

test_glm_chain: $(SRC)/test_glm_chain.cu $(HEADERS)
	$(NVCC) $(CFLAGS) -I$(SRC) -o $@ $< $(LDFLAGS)

test_glm_layer: $(SRC)/test_glm_layer.cu $(HEADERS)
	$(NVCC) $(CFLAGS) -I$(SRC) -o $@ $< $(LDFLAGS)

test_glm_expert_cache: $(SRC)/test_glm_expert_cache.cu $(HEADERS)
	$(NVCC) $(CFLAGS) -I$(SRC) -o $@ $< $(LDFLAGS)

test_glm_indexer_loader: $(SRC)/test_glm_indexer_loader.cu $(SRC)/glm_loader.cuh $(SRC)/safetensors_io.cuh
	$(NVCC) $(CFLAGS) -I$(SRC) -o $@ $< $(LDFLAGS)

test_glm_indexer: $(SRC)/test_glm_indexer.cu $(SRC)/glm_kernels.cuh $(SRC)/glm_primitives.cuh
	$(NVCC) $(CFLAGS) -I$(SRC) -o $@ $< $(LDFLAGS)

test_glm_mxfp4: $(SRC)/test_glm_mxfp4.cu $(SRC)/glm_kernels.cuh $(SRC)/glm_primitives.cuh
	$(NVCC) $(CFLAGS) -I$(SRC) -o $@ $< $(LDFLAGS)

tests: $(BINARIES)

clean:
	rm -f $(BINARIES)

.PHONY: all tests clean
