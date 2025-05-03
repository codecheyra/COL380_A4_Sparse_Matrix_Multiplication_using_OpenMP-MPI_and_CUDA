
#include <mpi.h>
#include <omp.h>
#include <vector>
#include <fstream>
#include <cassert>
#include <algorithm>
#include <cstdint>
#include <string>
#include <iostream>
// #include <chrono>
#include "sparse_matrix.hpp"

using ll = uint64_t;
// using Clock = std::chrono::high_resolution_clock;

extern "C" void multiplyTasksGPU(
    const ll *Aflat, int nA,
    const ll *Bflat, int nB,
    const int *tA, const int *tB,
    ll *Cout,
    int k, int KK, int tasksCount);

void SparseMatrix::load(const std::string &path)
{
    std::ifstream in(path);
    assert(in && "cannot open matrix file");
    in >> H >> W;
    n_blk_rows = (H + k - 1) / k;
    n_blk_cols = (W + k - 1) / k;

    int B;
    in >> B;
    blocks.resize(B);
    for (int i = 0; i < B; ++i)
    {
        int rawR, rawC;
        in >> rawR >> rawC;
        int rb = rawR / k, cb = rawC / k;
        blocks[i].row_blk = rb;
        blocks[i].col_blk = cb;

        int rows = std::min(k, H - rawR);
        int cols = std::min(k, W - rawC);
        auto &vals = blocks[i].vals;
        vals.assign((size_t)k * k, 0ULL);
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                in >> vals[r * k + c];
    }
}

void SparseMatrix::save(const std::string &path) const
{
    std::ofstream out(path);
    assert(out && "cannot open output file");
    out << H << " " << W << "\n"
        << blocks.size() << "\n";
    for (auto &b : blocks)
    {
        int rawR = b.row_blk * k, rawC = b.col_blk * k;
        out << rawR << " " << rawC << "\n";
        int rows = std::min(k, H - rawR),
            cols = std::min(k, W - rawC);
        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < cols; ++c)
                out << b.vals[r * k + c] << " ";
            out << "\n";
        }
    }
}

static SparseMatrix multiply(const SparseMatrix &A,
                             const SparseMatrix &B)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int k = A.k, KK = k * k;
    int R = A.n_blk_rows, S = B.n_blk_cols;
    int total = R * S;
    struct Task
    {
        int ia, ib, out;
    };
    std::vector<Task> tasks;
    tasks.reserve(A.blocks.size() * B.blocks.size());
    for (int ia = 0; ia < (int)A.blocks.size(); ++ia)
    {
        auto &a = A.blocks[ia];
        for (int ib = 0; ib < (int)B.blocks.size(); ++ib)
        {
            auto &b = B.blocks[ib];
            if (a.col_blk != b.row_blk)
                continue;
            int outp = a.row_blk * S + b.col_blk;
            tasks.push_back({ia, ib, outp});
        }
    }

    std::vector<Task> myT;
    myT.reserve((tasks.size() + size - 1) / size);
    for (size_t t = rank; t < tasks.size(); t += size)
        myT.push_back(tasks[t]);
    int Tn = (int)myT.size();

    int nA = (int)A.blocks.size(), nB = (int)B.blocks.size();
    std::vector<ll> Aflat((size_t)nA * KK), Bflat((size_t)nB * KK);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < nA; ++i)
    {
        std::copy(
            A.blocks[i].vals.begin(),
            A.blocks[i].vals.end(),
            Aflat.begin() + (size_t)i * KK);
    }
#pragma omp parallel for schedule(static)
    for (int i = 0; i < nB; ++i)
    {
        std::copy(
            B.blocks[i].vals.begin(),
            B.blocks[i].vals.end(),
            Bflat.begin() + (size_t)i * KK);
    }

    std::vector<int> tIA(Tn), tIB(Tn), tOut(Tn);
    for (int i = 0; i < Tn; ++i)
    {
        tIA[i] = myT[i].ia;
        tIB[i] = myT[i].ib;
        tOut[i] = myT[i].out;
    }

    std::vector<ll> Cpart((size_t)Tn * KK);
    multiplyTasksGPU(
        Aflat.data(), nA,
        Bflat.data(), nB,
        tIA.data(), tIB.data(),
        Cpart.data(),
        k, KK, Tn);

    std::vector<ll> acc((size_t)total * KK, 0ULL);
#pragma omp parallel for schedule(dynamic)
    for (int ti = 0; ti < Tn; ++ti)
    {
        int out = tOut[ti];
        size_t baseO = (size_t)out * KK, baseT = (size_t)ti * KK;
        for (int x = 0; x < KK; ++x)
        {
            acc[baseO + x] += Cpart[baseT + x];
        }
    }

    MPI_Allreduce(
        MPI_IN_PLACE,
        acc.data(),
        total * KK,
        MPI_UNSIGNED_LONG_LONG,
        MPI_SUM,
        MPI_COMM_WORLD);

    SparseMatrix C;
    C.H = A.H;
    C.W = B.W;
    C.k = k;
    C.n_blk_rows = R;
    C.n_blk_cols = S;
    for (int rb = 0; rb < R; ++rb)
    {
        for (int cb = 0; cb < S; ++cb)
        {
            int p = rb * S + cb;
            ll *ptr = acc.data() + (size_t)p * KK;
            bool any = false;
            for (int i = 0; i < KK; ++i)
            {
                if (ptr[i] != 0ULL)
                {
                    any = true;
                    break;
                }
            }
            if (!any)
                continue;
            C.blocks.push_back({rb, cb,
                                std::vector<ll>(ptr, ptr + KK)});
        }
    }
    return C;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    // auto t0 = Clock::now();

    if (argc != 2)
    {
        if (MPI::COMM_WORLD.Get_rank() == 0)
            std::cerr << "Usage: " << argv[0] << " <dir>\n";
        MPI_Finalize();
        return 1;
    }
    std::string dir = argv[1];

    int N, k;
    {
        std::ifstream in(dir + "/size");
        assert(in && "cannot open size");
        in >> N >> k;
    }
    // auto t1 = Clock::now();

    std::vector<SparseMatrix> M(N);
    for (int i = 0; i < N; ++i)
    {
        M[i].k = k;
        M[i].load(dir + "/matrix" + std::to_string(i + 1));
    }
    // auto t2 = Clock::now();

    while (M.size() > 1)
    {
        int best = 0;
        size_t cost = M[0].blocks.size() * M[1].blocks.size();
        for (int i = 1; i + 1 < (int)M.size(); ++i)
        {
            size_t c = M[i].blocks.size() * M[i + 1].blocks.size();
            if (c < cost)
                cost = c, best = i;
        }
        M[best] = multiply(M[best], M[best + 1]);
        M.erase(M.begin() + best + 1);
    }
    // auto t3 = Clock::now();

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
        M[0].save("matrix");
    // auto t4 = Clock::now();

    // if (rank == 0)
    // {
    //     using sec = std::chrono::duration<double>;
    //     std::cout
    //         << "Timings (s): read=" << sec(t1 - t0).count()
    //         << " load=" << sec(t2 - t1).count()
    //         << " compute=" << sec(t3 - t2).count()
    //         << " save=" << sec(t4 - t3).count()
    //         << " total=" << sec(t4 - t0).count()
    //         << "\n";
    // }
    MPI_Finalize();
    return 0;
}