/*!
 * \file tl/maca/op/scan.cc
 * \brief MACA implementation registration for tl scan lowering.
 */

#include "backend/common/op/scan.h"

#include "backend/common/target_utils.h"

namespace tvm {
namespace tl {

namespace {

bool MatchMacaScanTarget(Target target) {
  return TargetIsMaca(target) || TargetIsCuTeDSL(target);
}

bool RegisterMacaScan() {
  RegisterCumSumImpl(CumSumImpl{
      "maca.CumSum",
      MatchMacaScanTarget,
      backend::scan::LowerCumSum,
  });
  RegisterCumMaxImpl(CumMaxImpl{
      "maca.CumMax",
      MatchMacaScanTarget,
      backend::scan::LowerCumMax,
  });
  return true;
}

const bool maca_scan_registered = RegisterMacaScan();

} // namespace

} // namespace tl
} // namespace tvm
