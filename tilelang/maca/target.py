from __future__ import annotations

from tvm.target import Target

from tilelang.backend.target import register_target_detector


def _target_ffi_api():
    from tilelang import _ffi_api

    return _ffi_api


def check_maca_availability() -> bool:
    """
    Check if MACA is available on the system by locating the MACA path.
    Returns:
        bool: True if MACA is available, False otherwise.
    """
    try:
        from tilelang.contrib import mxcc

        mxcc.find_maca_path()
        return True
    except Exception:
        return False


def _detect_maca_target() -> Target | str | None:
    import torch

    if torch.version.hip is not None:
        return None
    if not check_maca_availability():
        return None

    return Target("maca")


def target_is_maca(target: Target) -> bool:
    return _target_ffi_api().TargetIsMaca(target)


def target_has_async_copy(target: Target) -> bool:
    return _target_ffi_api().TargetHasAsyncCopy(target)


register_target_detector("maca", _detect_maca_target, override=True)
