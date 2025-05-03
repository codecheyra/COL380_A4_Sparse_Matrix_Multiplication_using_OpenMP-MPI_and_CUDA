// File: a4.cpp

#include <mpi.h>
#include <vector>
#include <map>
#include <algorithm>
#include <fstream>
#include <cassert>
#include <cstdint>
#include <string>
#include "sparse_matrix.hpp"

// ——— load() reads each block’s coords + k×k (or smaller) values in the
//     same interleaved order your testcases use ———
void SparseMatrix::load(const std::string& path) {
    std::ifstream in(path);
    assert(in && "cannot open matrix file");
    in >> H >> W;
    n_blk_rows = (H + k - 1) / k;
    n_blk_cols = (W + k - 1) / k;

    int B; in >> B;
    blocks.resize(B);
    for (int i = 0; i < B; ++i) {
        int rawR, rawC;
        in >> rawR >> rawC;
        int rb = rawR / k;
        int cb = rawC / k;
        blocks[i].row_blk = rb;
        blocks[i].col_blk = cb;

        // edge‐blocks may be smaller
        int rows = std::min(k, H - rawR);
        int cols = std::min(k, W - rawC);

        blocks[i].vals.assign(k*k, 0ULL);
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                in >> blocks[i].vals[r*k + c];
    }
}

// ——— save() writes element‐offsets = block_index * k, then exactly
//     rows×cols lines of values ———
void SparseMatrix::save(const std::string& path) const {
    std::ofstream out(path);
    assert(out && "cannot open output file");

    // 1) dimensions
    out << H << " " << W << "\n";
    // 2) non‑zero block count
    out << blocks.size() << "\n";
    // 3) for each block:
    for (auto &b : blocks) {
        int rawR = b.row_blk * k;
        int rawC = b.col_blk * k;
        out << rawR << " " << rawC << "\n";

        int rows = std::min(k, H - rawR);
        int cols = std::min(k, W - rawC);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c)
                out << b.vals[r*k + c] << " ";
            out << "\n";
        }
    }
}

// ——— pure‑CPU multiply A×B into a new SparseMatrix ———
static SparseMatrix multiply(const SparseMatrix &A,
                             const SparseMatrix &B) {
    assert(A.k == B.k);
    int k = A.k, KK = k*k;
    SparseMatrix C;
    C.H = A.H;  C.W = B.W;  C.k = k;
    C.n_blk_rows = A.n_blk_rows;
    C.n_blk_cols = B.n_blk_cols;

    // accumulate each C‑block in a map keyed by (rb, cb)
    std::map<std::pair<int,int>, std::vector<ll>> acc;
    for (auto &a : A.blocks) {
        for (auto &b : B.blocks) {
            if (a.col_blk != b.row_blk) continue;
            int rb = a.row_blk, cb = b.col_blk;
            auto &blk = acc[{rb, cb}];
            if (blk.empty()) blk.assign(KK, 0ULL);

            int rows   = std::min(k, A.H - rb*k),
                cols   = std::min(k, B.W - cb*k),
                common = std::min(k, A.W - a.col_blk*k);

            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    ll sum = 0;
                    for (int t = 0; t < common; ++t)
                        sum += a.vals[i*k + t] * b.vals[t*k + j];
                    blk[i*k + j] += sum;
                }
            }
        }
    }

    // move only non‑zero blocks into C
    for (auto &p : acc) {
        auto &vals = p.second;
        bool any = std::any_of(vals.begin(), vals.end(),
                               [](ll v){ return v != 0; });
        if (!any) continue;
        C.blocks.push_back({p.first.first,
                            p.first.second,
                            std::move(vals)});
    }
    return C;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <testdir>\n";
        MPI_Finalize();
        return 1;
    }
    std::string dir = argv[1];

    // 1) read N,k
    int N,k;
    {
        std::ifstream in(dir + "/size");
        assert(in && "cannot open size");
        in >> N >> k;
    }

    // 2) load all N matrices
    std::vector<SparseMatrix> M(N);
    for (int i = 0; i < N; ++i) {
        M[i].k = k;
        M[i].load(dir + "/matrix" + std::to_string(i+1));
    }

    // 3) greedy chain multiply
    while (M.size() > 1) {
        int best = 0;
        size_t cost = M[0].blocks.size() * M[1].blocks.size();
        for (int i = 1; i+1 < (int)M.size(); ++i) {
            size_t c = M[i].blocks.size() * M[i+1].blocks.size();
            if (c < cost) {
                cost = c;
                best = i;
            }
        }
        M[best] = multiply(M[best], M[best+1]);
        M.erase(M.begin() + best + 1);
    }

    // 4) save result as “matrix”
    M[0].save("matrix");

    MPI_Finalize();
    return 0;
}
