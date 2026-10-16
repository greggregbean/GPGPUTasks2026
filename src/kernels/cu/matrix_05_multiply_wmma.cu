#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

// Include WMMA header with nvcuda::wmma namespace
// Если строка "using namespace nvcuda;" не компилируется, добавьте в CMake options: -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_FLAGS=-lineinfo
#include <mma.h>
using namespace nvcuda;

__global__ void matrix_multiply_wmma(
    const float* a, // rows=h x cols=k
    const float* b, // rows=k x cols=w
    float* c, // rows=h x cols=w
    unsigned int M,
    unsigned int N,
    unsigned int K)
{
    constexpr unsigned m_tile_sz = 16;
    constexpr unsigned n_tile_sz = 16;
    constexpr unsigned k_tile_sz = 16;

    const unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned warp_id = tid / 32;

    wmma::fragment<wmma::matrix_a, m_tile_sz, n_tile_sz, k_tile_sz, __half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, m_tile_sz, n_tile_sz, k_tile_sz, __half, wmma::row_major> frag_b;
    wmma::fragment<wmma::accumulator, m_tile_sz, n_tile_sz, k_tile_sz, float> frag_c;

    if (warp_id == 0)
        wmma::fill_fragment(frag_c, 0.0f);

    __shared__ __half __align__(32) loc_a[m_tile_sz][k_tile_sz];
    __shared__ __half __align__(32) loc_b[k_tile_sz][n_tile_sz];
    __shared__ float __align__(32) loc_c[m_tile_sz][n_tile_sz];

    unsigned num_tiles = (K + k_tile_sz - 1) / k_tile_sz;

    unsigned glob_a_i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned glob_b_j = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned t = 0; t < num_tiles; t++) {
        // Load A
        unsigned glob_a_j = t * k_tile_sz + threadIdx.x;

        float fp32_v = 0;

        if (glob_a_i < M && glob_a_j < K)
            fp32_v = a[glob_a_i * K + glob_a_j];

        unsigned loc_a_j = threadIdx.x;
        unsigned loc_a_i = threadIdx.y;

        loc_a[loc_a_i][loc_a_j] = __float2half_rn(fp32_v);

        // Load B
        unsigned glob_b_i = t * k_tile_sz + threadIdx.y;

        fp32_v = 0;

        if (glob_b_i < K && glob_b_j < N)
            fp32_v = b[glob_b_i * N + glob_b_j];

        unsigned loc_b_j = threadIdx.x;
        unsigned loc_b_i = threadIdx.y;

        loc_b[loc_b_i][loc_b_j] = __float2half_rn(fp32_v);

        __syncthreads();

        // MMA
        if (warp_id == 0) {
            wmma::load_matrix_sync(frag_a, &loc_a[0][0], k_tile_sz);
            wmma::load_matrix_sync(frag_b, &loc_b[0][0], n_tile_sz);
            wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        }

        __syncthreads();
    }

    // Store C
    if (warp_id == 0)
        wmma::store_matrix_sync(&loc_c[0][0], frag_c, n_tile_sz, wmma::mem_row_major);

    __syncthreads();

    const unsigned glob_c_j = blockIdx.x * n_tile_sz + threadIdx.x;
    const unsigned glob_c_i = blockIdx.y * m_tile_sz + threadIdx.y;

    if (glob_c_i < M && glob_c_j < N)
        c[glob_c_j + glob_c_i * N] = loc_c[threadIdx.y][threadIdx.x];
}

namespace cuda {
void matrix_multiply_wmma(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_wmma<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), h, w, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
