CXX       = mpiCC
NVCC      = nvcc

CXXFLAGS  = -O3 -fopenmp -std=c++14
NVCCFLAGS = -O3 -std=c++11
LDFLAGS   = -lcudart

OBJS      = a4.o cuda_kernels.o
TARGET    = a4

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

a4.o: a4.cpp sparse_matrix.hpp
	$(CXX) $(CXXFLAGS) -c a4.cpp -o a4.o

cuda_kernels.o: cuda_kernels.cu sparse_matrix.hpp
	$(NVCC) $(NVCCFLAGS) -c cuda_kernels.cu -o cuda_kernels.o

clean:
	rm -f $(TARGET) *.o
