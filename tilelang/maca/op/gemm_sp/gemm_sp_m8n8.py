"""MACA gemm_sp path for M/N that are multiples of 8 but not both multiples of 16.

No 16x8 MMA builtin: densify sparse A into shared, then SIMT FMA.
"""

from __future__ import annotations

from tvm import DataType
from tvm.target import Target
from tvm import tirx
from tvm.tirx import BufferRegion

from tilelang import language as T
from tilelang.layout import make_swizzled_layout
from tilelang.tileop.gemm_sp.gemm_sp_base import GemmSPBase
from tilelang.transform.simplify import _Simplify
from tilelang.utils.language import is_fragment


GEMM_SP_INST_M8N8 = "maca.mma.sp.m8n8"


def _simt_layout(shape, threads: int) -> T.Fragment:
    extent = 1
    for s in shape:
        extent *= int(s)
    assert extent % threads == 0, f"shape {shape} not divisible by threads {threads}"

    def forward_thread(*idx):
        linear = 0
        for i, s in zip(idx, shape):
            linear = linear * int(s) + i
        return linear % threads

    def forward_index(*idx):
        linear = 0
        for i, s in zip(idx, shape):
            linear = linear * int(s) + i
        return linear // threads

    return T.Fragment(list(shape), forward_thread_fn=forward_thread, forward_index_fn=forward_index)


def _region2d(region: BufferRegion):
    """Split a BufferRegion into physical buffer + leading mins + last-two bases."""
    buf = region.buffer
    other = [r.min for r in region.region[:-2]]
    b0 = region.region[-2].min
    b1 = region.region[-1].min
    return buf, other, b0, b1


class GemmSPM8N8(GemmSPBase):
    def infer_layout(self, target: Target, thread_nums: int):
        layouts = {self.C: _simt_layout(self.C.shape, thread_nums)}
        if is_fragment(self.A):
            layouts[self.A] = _simt_layout(self.A.shape, thread_nums)
        else:
            layouts[self.A] = make_swizzled_layout(self.A)
        if is_fragment(self.B):
            layouts[self.B] = _simt_layout(self.B.shape, thread_nums)
        else:
            layouts[self.B] = make_swizzled_layout(self.B)
        return layouts

    def lower(self, layout_map: dict, target: Target, thread_bounds: range, thread_var: tirx.Var):
        thread_nums = int(thread_bounds.extent)
        M, N, K = int(self.M), int(self.N), int(self.K)
        assert M * N % thread_nums == 0
        assert K % 4 == 0, "m8n8 SIMT densify currently supports 2:4 sparsity (K % 4 == 0)"

        in_dtype = self.a_dtype
        e_bits = DataType(self.e_dtype).bits
        trans_A, trans_E, trans_B = self.trans_A, self.trans_E, self.trans_B
        clear_accum = self.clear_accum
        C = self.C
        accum_dtype = self.accum_dtype
        b_is_frag = is_fragment(self.B)

        A_buf, A_other, A_b0, A_b1 = _region2d(self.ARegion)
        E_buf, E_other, E_b0, E_b1 = _region2d(self.ERegion)
        B_buf, B_other, B_b0, B_b1 = _region2d(self.BRegion)

        layout_mk = _simt_layout((M, K), thread_nums)
        layout_m_ks = _simt_layout((M, K // 2), thread_nums)
        layout_ks_m = _simt_layout((K // 2, M), thread_nums)
        layout_nk = _simt_layout((N, K), thread_nums)
        layout_kn = _simt_layout((K, N), thread_nums)
        layout_mn = _simt_layout((M, N), thread_nums)

        @T.prim_func
        def _gemm_m8n8() -> None:
            A_dense = T.alloc_shared((M, K), in_dtype)

            for i, k in T.Parallel(M, K, loop_layout=layout_mk):
                A_dense[i, k] = 0
            T.sync_threads()

            # Scatter 2:4 sparse A (+E) into dense shared (region-aware for pipeline stages).
            if trans_A:
                for ks, i in T.Parallel(K // 2, M, loop_layout=layout_ks_m):
                    g = ks // 2
                    bit_off = g * 4
                    e_col = bit_off // e_bits
                    if trans_E:
                        e_val = E_buf[tuple(E_other) + (E_b0 + e_col, E_b1 + i)]
                    else:
                        e_val = E_buf[tuple(E_other) + (E_b0 + i, E_b1 + e_col)]
                    meta = (e_val >> (bit_off % e_bits)) & 0xF
                    idx = (meta >> ((ks % 2) * 2)) & 0x3
                    A_dense[i, g * 4 + idx] = A_buf[tuple(A_other) + (A_b0 + ks, A_b1 + i)]
            else:
                for i, ks in T.Parallel(M, K // 2, loop_layout=layout_m_ks):
                    g = ks // 2
                    bit_off = g * 4
                    e_col = bit_off // e_bits
                    if trans_E:
                        e_val = E_buf[tuple(E_other) + (E_b0 + e_col, E_b1 + i)]
                    else:
                        e_val = E_buf[tuple(E_other) + (E_b0 + i, E_b1 + e_col)]
                    meta = (e_val >> (bit_off % e_bits)) & 0xF
                    idx = (meta >> ((ks % 2) * 2)) & 0x3
                    A_dense[i, g * 4 + idx] = A_buf[tuple(A_other) + (A_b0 + i, A_b1 + ks)]
            T.sync_threads()

            if b_is_frag:
                if trans_B:
                    B_s = T.alloc_shared((N, K), in_dtype)
                    for j, k in T.Parallel(N, K, loop_layout=layout_nk):
                        B_s[j, k] = B_buf[tuple(B_other) + (B_b0 + j, B_b1 + k)]
                else:
                    B_s = T.alloc_shared((K, N), in_dtype)
                    for k, j in T.Parallel(K, N, loop_layout=layout_kn):
                        B_s[k, j] = B_buf[tuple(B_other) + (B_b0 + k, B_b1 + j)]
                T.sync_threads()

                if clear_accum:
                    T.clear(C)

                for i, j in T.Parallel(M, N, loop_layout=layout_mn):
                    for k in T.serial(K):
                        b_val = B_s[j, k] if trans_B else B_s[k, j]
                        C[i, j] = C[i, j] + T.cast(A_dense[i, k], accum_dtype) * T.cast(b_val, accum_dtype)
            else:
                if clear_accum:
                    T.clear(C)

                for i, j in T.Parallel(M, N, loop_layout=layout_mn):
                    for k in T.serial(K):
                        if trans_B:
                            b_val = B_buf[tuple(B_other) + (B_b0 + j, B_b1 + k)]
                        else:
                            b_val = B_buf[tuple(B_other) + (B_b0 + k, B_b1 + j)]
                        C[i, j] = C[i, j] + T.cast(A_dense[i, k], accum_dtype) * T.cast(b_val, accum_dtype)

        return _Simplify(_gemm_m8n8, inline_let=True)
