SHELL := /bin/bash

IDIR = include
SDIR = src
SCUDADIR = src_cuda
ODIR = obj
BINDIR = bin
OUT_DIR = output
DATADIR = data

SUBDIRS = $(ODIR) $(BINDIR) $(DATADIR) $(OUT_DIR) 

USE_CUDA = 0
USE_OMP = 1
USE_NCCL = 1
O_LEVEL = 3


# check if flag is active and nvcc is installed
ifeq ($(USE_CUDA), 1)
	ifeq (, $(shell which nvcc))
		$(error "nvcc is not installed, please install it to use CUDA")
	endif
	ifeq ($(USE_NCCL), 1)
 		NCCLDFLAGS := -lnccl
 	endif
	CXX := nvcc
	CUFLAGS := -arch=sm_80 -Xcompiler -Wall -Xcompiler -Wextra -G
	CULDFLAGS := -lcuda $(NCCLDFLAGS) -lcublas
else
	CXX := g++
	CROSSCOMPILECU := -x c++
endif

CXXFLAGS = -I$(IDIR) -I$(SCUDADIR) -std=c++17 -g -O$(O_LEVEL) $(CUFLAGS) 
CPPFLAGS = -DUSE_CUDA=$(USE_CUDA) -DUSE_OMP=$(USE_OMP) -DUSE_NCCL=$(USE_NCCL)
LDFLAGS = $(CULDFLAGS)

ifeq ($(USE_OMP), 1)
	ifeq ($(USE_CUDA), 1)
		CXXFLAGS += -Xcompiler -fopenmp -Xcompiler -lmpi
	else
		CXXFLAGS += -fopenmp
	endif
	LDFLAGS += -lgomp -lmpi
endif

# DEPS = $(IDIR)/$(wildcard *.hpp *.cuh) $(SCUDADIR)/$(wildcard *.cu)

_CXXFILES = $(wildcard $(SDIR)/*.cpp) $(wildcard $(SDIR)/*.cu)
CXXFILES = $(notdir $(_CXXFILES))

_DEPS = $(filter %.d, $(CXXFILES:.cpp=.d) $(CXXFILES:.cu=.d))
DEPS = $(patsubst %,$(ODIR)/%,$(_DEPS))

_OBJ = $(filter-out main%, $(filter %.o, $(CXXFILES:.cpp=.o) $(CXXFILES:.cu=.o)))
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

ifeq ($(USE_NCCL), 1)
	OBJ += $(ODIR)/main_nccl.o
else
	OBJ += $(ODIR)/main.o
endif

TARGET = $(BINDIR)/conj

UNAME_S := $(shell uname -s)

$(TARGET): $(OBJ) | subdirs
	$(CXX) -o $@ $^ $(LDFLAGS)

run: $(TARGET)
	./$(TARGET) $(DATADIR)/mtx.bin $(DATADIR)/rhs.bin $(OUT_DIR)/solution.bin 1000 1e-16

all: $(TARGET) run

$(ODIR)/%.o: $(SDIR)/%.cpp Makefile | $(ODIR)
	$(CXX) -c -o $@ $< $(CXXFLAGS) $(CPPFLAGS) -MMD -MP

$(ODIR)/%.o: $(SDIR)/%.cu Makefile | $(ODIR)
	$(CXX) -c -o $@ $(CROSSCOMPILECU) $< $(CXXFLAGS) $(CPPFLAGS) -MMD -MP

-include $(DEPS)

.PHONY: clean run subdirs

clean:
	rm -f $(ODIR)/*.o  $(ODIR)/*.d $(TARGET)

subdirs: | $(SUBDIRS)

$(SUBDIRS):
	mkdir -p $@