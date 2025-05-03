#ifndef SPARSE_MATRIX_HPP
#define SPARSE_MATRIX_HPP

#include <vector>
#include <string>
#include <cstdint>
// Use unsigned 64-bit for element values and accumulations
using ll = uint64_t;

struct Block {
    int row_blk, col_blk;
    std::vector<ll> vals;  // flattened k×k block, row-major
};

struct SparseMatrix {
    int H, W, k;
    int n_blk_rows, n_blk_cols;
    std::vector<Block> blocks;

    // Load a sparse-blocked matrix from “path”
    void load(const std::string& path);

    // Save this matrix to “path”
    void save(const std::string& path) const;
};

// Compute one k×k block C += A×B on the GPU.
//   a, b: input blocks (size k*k), c: output block (size k*k)
//   rows, cols account for edge-blocks smaller than k.
void multiplyBlockGPU(const ll* a, const ll* b, ll* c,
                      int k, int rows, int cols);

#endif // SPARSE_MATRIX_HPP
