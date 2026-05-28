# Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.

from __future__ import annotations
import tvm
from tvm.target import Target
from .arch_base import TileDevice
from .driver import maca_driver


def is_maca_arch(arch: TileDevice) -> bool:
    return isinstance(arch, MACA)


def _get_target_attr(target: Target, name: str) -> int | None:
    value = getattr(target, name, None)
    if value is None:
        value = target.attrs.get(name, None)
    return int(value) if value is not None else None


def _get_l2_cache_size_bytes(target: Target) -> int:
    target_value = _get_target_attr(target, "l2_cache_size_bytes")
    if target_value is not None:
        return target_value

    prop = maca_driver.get_maca_device_properties()
    if prop is not None and hasattr(prop, "L2_cache_size"):
        return int(prop.L2_cache_size)

    persisting_l2 = maca_driver.get_persisting_l2_cache_max_size()
    return int(persisting_l2) if persisting_l2 is not None else 0


class TensorInstruction:
    def __init__(self, name: str, shape: list[int]):
        self.name: str = name
        # only hold the shape of M and N
        self.shape: list[int] = shape


class MACA(TileDevice):
    # FIXME: config should meets MACA
    def __init__(self, target: Target | str):
        if isinstance(target, str):
            target = tvm.target.Target(target)
        self.target = target
        device = tvm.device(tvm.ffi.DLDeviceType.kDLMACA, 0)
        if not device.exist:
            raise RuntimeError("Cannot find MACA device 0.")
        self.device: tvm.runtime.Device = device
        self.platform: str = "MACA"
        self.smem_cap = device.max_shared_memory_per_block
        self.compute_max_core = device.multi_processor_count
        self.warp_size = device.warp_size
        self.compute_capability = device.compute_version.replace(".", "")
        self.reg_cap: int = 65536
        self.max_smem_usage: int = 2 * self.smem_cap
        self.sm_partition: int = 8
        self.l2_cache_size_bytes: int = _get_l2_cache_size_bytes(target)
        self.transaction_size: list[int] = [32, 128]  # in bytes

        self.bandwidth: list[int] = [750, 12080]

    def get_avaliable_tensorintrin_shapes(self):
        self.available_tensor_instructions = (TensorInstruction("mma", [16, 16]),)
        return [t.shape for t in self.available_tensor_instructions]
