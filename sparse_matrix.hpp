#ifndef SPARSE_MATRIX_HPP
#define SPARSE_MATRIX_HPP

#include <vector>
#include <string>
#include <cstdint>
using ll = uint64_t;

struct Block {
    int row_blk, col_blk;
    std::vector<ll> vals;
};

struct SparseMatrix {
    int H, W, k;
    int n_blk_rows, n_blk_cols;
    std::vector<Block> blocks;

    void load(const std::string& path);
    void save(const std::string& path) const;
};

extern "C" void multiplyTasksGPU(
    const ll* Aflat, int nA,
    const ll* Bflat, int nB,
    const int* tA,  const int* tB,
    ll* Cout,
    int k, int KK, int tasksCount
);

#endif
