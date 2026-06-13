from __future__ import annotations

from tvm.target import Target

from tilelang.backend.execution_backend import ExecutionBackendSpec, register_execution_backend


def _is_cutedsl_target(target: Target) -> bool:
    return target.kind.name == "maca" and "cutedsl" in target.keys


def _is_plain_maca_target(target: Target) -> bool:
    return target.kind.name == "maca" and "cutedsl" not in target.keys


def _is_mcrtc_available() -> bool:
    try:
        from tilelang.jit.adapter.mcrtc import is_mcrtc_available
    except ImportError:
        return False
    return bool(is_mcrtc_available)


def _is_cutedsl_available() -> bool:
    try:
        from tilelang.jit.adapter.cutedsl.checks import check_cutedsl_available

        check_cutedsl_available()
    except ImportError:
        return False
    return True


register_execution_backend(
    "maca",
    ExecutionBackendSpec(
        "tvm_ffi",
        supports_target=_is_plain_maca_target,
        enable_host_codegen=True,
        enable_device_compile=True,
    ),
    override=True,
)
register_execution_backend(
    "maca",
    ExecutionBackendSpec("mcrtc", is_available=_is_mcrtc_available, supports_target=_is_plain_maca_target),
    override=True,
)
register_execution_backend(
    "maca",
    ExecutionBackendSpec("cython", supports_target=_is_plain_maca_target),
    override=True,
)
register_execution_backend(
    "maca",
    ExecutionBackendSpec("cutedsl", is_available=_is_cutedsl_available, supports_target=_is_cutedsl_target),
    override=True,
)
