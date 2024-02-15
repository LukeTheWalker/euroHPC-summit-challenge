SHELL := /bin/bash

IDIR = include
SDIR = src
SCUDADIR = src_cuda
ODIR = obj
BINDIR = bin
OUT_DIR = output
DATADIR = data

SUBDIRS = $(ODIR) $(BINDIR) $(DATADIR) $(OUT_DIR) 

USE_CUDA = 1
USE_OMP = 1

O_LEVEL = 3


# check if flag is active and nvcc is installed
ifeq ($(USE_CUDA), 1)
	ifeq (, $(shell which nvcc))
		$(error "nvcc is not installed, please install it to use CUDA")
	endif
	CXX := nvcc
	CUFLAGS := -arch=sm_80 -Xcompiler -Wall -Xcompiler -Wextra 
	CULDFLAGS := -lcuda
else
	CXX := g++
endif

CXXFLAGS = -I$(IDIR) -I$(SCUDADIR) -std=c++17 -g -O$(O_LEVEL) $(CUFLAGS) -DUSE_CUDA=$(USE_CUDA) -DUSE_OMP=$(USE_OMP) 
LDFLAGS = $(CULDFLAGS)

ifeq ($(USE_OMP), 1)
	ifeq ($(USE_CUDA), 1)
		CXXFLAGS += -Xcompiler -fopenmp
	else
		CXXFLAGS += -fopenmp
	endif
	LDFLAGS += -lgomp
endif

DEPS = $(IDIR)/$(wildcard *.hpp *.cuh) $(SCUDADIR)/$(wildcard *.cu)

_CXXFILES = $(wildcard $(SDIR)/*.cpp) $(wildcard $(SDIR)/*.cu)

CXXFILES = $(notdir $(_CXXFILES))
 
_OBJ = $(CXXFILES:.cpp=.o)
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ:.cu=.o))

TARGET = $(BINDIR)/conj

UNAME_S := $(shell uname -s)

$(TARGET): $(OBJ) | subdirs
	$(CXX) -o $@ $^ $(LDFLAGS)

run: $(TARGET)
	./$(TARGET) $(DATADIR)/mtx.bin $(DATADIR)/rhs.bin $(OUT_DIR)/solution.bin 1000 1e-16

all: $(TARGET) run

$(ODIR)/%.o: $(SDIR)/%.cpp $(DEPS) | $(ODIR)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

$(ODIR)/%.o: $(SDIR)/%.cu $(DEPS) | $(ODIR)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

.PHONY: clean run subdirs

clean:
	rm -f $(ODIR)/*.o $(TARGET)

subdirs: | $(SUBDIRS)

$(SUBDIRS):
	mkdir -p $@