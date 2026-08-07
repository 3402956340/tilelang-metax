import tilelang
import tilelang.language as T
import tilelang.testing
import example_gemm
import example_gemm_intrinsics

_BENCH_GEMM_CONFIG = {
    "block_M": 128,
    "block_N": 128,
    "block_K": 128,
    "threads": 256,
    "num_stages": 0,
}

_BENCH_GEMM_CASES = (
    {"name": "bench_gemm_m1664_n1024_k262144", "M": 1664, "N": 1024, "K": 262144},
    {"name": "bench_gemm_m4096_n1024_k8192", "M": 4096, "N": 1024, "K": 8192},
    {"name": "bench_gemm_m4096_n8192_k8192", "M": 4096, "N": 8192, "K": 8192},
    {"name": "bench_gemm_m4096_n28672_k8192", "M": 4096, "N": 28672, "K": 8192},
    {"name": "bench_gemm_m4096_n8192_k28672", "M": 4096, "N": 8192, "K": 28672},
    {"name": "bench_gemm_m8192_n1024_k8192", "M": 8192, "N": 1024, "K": 8192},
    {"name": "bench_gemm_m8192_n8192_k8192", "M": 8192, "N": 8192, "K": 8192},
    {"name": "bench_gemm_m8192_n28672_k8192", "M": 8192, "N": 28672, "K": 8192},
    {"name": "bench_gemm_m8192_n8192_k28672", "M": 8192, "N": 8192, "K": 28672},
)


@tilelang.jit(out_idx=[-1])
def _bench_gemm_matmul(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    threads,
    num_stages,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.use_swizzle(panel_size=10)
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


def _run_bench_gemm(M, N, K, block_M, block_N, block_K, threads, num_stages):
    kernel = _bench_gemm_matmul(M, N, K, block_M, block_N, block_K, threads, num_stages)
    profiler = kernel.get_profiler()
    return profiler.do_bench(backend="cupti")


def _process_bench_gemm_case(case):
    tilelang.testing.process_func(
        _run_bench_gemm,
        case["name"],
        M=case["M"],
        N=case["N"],
        K=case["K"],
        **_BENCH_GEMM_CONFIG,
    )


def _get_bench_gemm_case(name):
    for case in _BENCH_GEMM_CASES:
        if case["name"] == name:
            return case
    raise KeyError(f"unknown GEMM benchmark case: {name}")


def regression_bench_gemm_m1664_n1024_k262144():
    _process_bench_gemm_case(_get_bench_gemm_case("bench_gemm_m1664_n1024_k262144"))


def regression_bench_gemm_m4096_n1024_k8192():
    _process_bench_gemm_case(_get_bench_gemm_case("bench_gemm_m4096_n1024_k8192"))


def regression_bench_gemm_m4096_n8192_k8192():
    _process_bench_gemm_case(_get_bench_gemm_case("bench_gemm_m4096_n8192_k8192"))


def regression_bench_gemm_m4096_n28672_k8192():
    _process_bench_gemm_case(_get_bench_gemm_case("bench_gemm_m4096_n28672_k8192"))


def regression_bench_gemm_m4096_n8192_k28672():
    _process_bench_gemm_case(_get_bench_gemm_case("bench_gemm_m4096_n8192_k28672"))


def regression_bench_gemm_m8192_n1024_k8192():
    _process_bench_gemm_case(_get_bench_gemm_case("bench_gemm_m8192_n1024_k8192"))


def regression_bench_gemm_m8192_n8192_k8192():
    _process_bench_gemm_case(_get_bench_gemm_case("bench_gemm_m8192_n8192_k8192"))


def regression_bench_gemm_m8192_n28672_k8192():
    _process_bench_gemm_case(_get_bench_gemm_case("bench_gemm_m8192_n28672_k8192"))


def regression_bench_gemm_m8192_n8192_k28672():
    _process_bench_gemm_case(_get_bench_gemm_case("bench_gemm_m8192_n8192_k28672"))


def regression_example_gemm_intrinsics():
    tilelang.testing.process_func(example_gemm_intrinsics.run_regression_perf, M=1024, N=1024, K=1024)


def regression_example_gemm():
    tilelang.testing.process_func(example_gemm.run_regression_perf)


if __name__ == "__main__":
    tilelang.testing.regression()
