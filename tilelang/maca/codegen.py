from __future__ import annotations

from tvm.target import Target

from tilelang.backend.device_codegen import DeviceCodegen, global_func_device_codegen, register_device_codegen


def _is_plain_maca_target(target: Target) -> bool:
    return target.kind.name == "maca"


register_device_codegen(
    "maca",
    DeviceCodegen(
        "maca",
        build=global_func_device_codegen("target.build.tilelang_maca"),
        build_without_compile=global_func_device_codegen("target.build.tilelang_maca_without_compile"),
        supports_target=_is_plain_maca_target,
    ),
    override=True,
)
