"""MACA sparse GEMM op registrations."""

from __future__ import annotations

from tilelang.tileop.gemm_sp.registry import register_gemm_sp_impl
from tilelang.maca.op.gemm_sp.gemm_sp_mma import GEMM_SP_INST_MMA_SP, GemmSPMMA
from tilelang.maca.op.gemm_sp.gemm_sp_m8n8 import GEMM_SP_INST_M8N8, GemmSPM8N8
from tilelang.maca.target import target_is_maca


register_gemm_sp_impl("maca.GemmSPMMA", GEMM_SP_INST_MMA_SP, target_is_maca, GemmSPMMA)
register_gemm_sp_impl("maca.GemmSPM8N8", GEMM_SP_INST_M8N8, target_is_maca, GemmSPM8N8)
