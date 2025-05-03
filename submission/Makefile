# File: Makefile

# Load MPI+CXX wrapper and CUDA compiler
CXX      = mpiCC
NVCC     = nvcc

# Compiler flags
CXXFLAGS = -O3 -fopenmp
NVCCFLAGS= -O3 -std=c++11

# Linker flags
LDFLAGS  = -lcudart

# Objects and target
OBJS     = a4.o cuda_kernels.o
TARGET   = a4

# Default target
all: $(TARGET)

# Link step
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

# Compile MPI+OpenMP C++ code
a4.o: a4.cpp sparse_matrix.hpp
	$(CXX) $(CXXFLAGS) -c a4.cpp

# Compile CUDA kernels
cuda_kernels.o: cuda_kernels.cu sparse_matrix.hpp
	$(NVCC) $(NVCCFLAGS) -c cuda_kernels.cu -o cuda_kernels.o

# Clean up
clean:
	rm -f $(TARGET) *.o
