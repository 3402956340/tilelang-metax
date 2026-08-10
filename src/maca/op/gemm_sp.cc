/*!
 * \file tl/backend/maca/op/gemm_sp.cc
 * \brief MACA implementation for tl.gemm_sp instruction selection.
 */
#include "op/gemm_sp.h"
#include "support/check.h"

#include "maca/target_utils.h"

#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/transform.h>

#include <cmath>
#include <limits>
#include <utility>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace maca {

namespace {

constexpr const char *kMacaMMASP = "maca.mma.sp";
constexpr const char *kMacaMMASPM8N8 = "maca.mma.sp.m8n8";

std::pair<int, int>
ComputeDefaultWarpPartition(const GemmSPWarpPolicyNode &policy, int M, int N,
                            int num_warps, int k_m_per_warp, int k_n_per_warp) {
  int m_warp = 1, n_warp = 1;

  ICHECK(M % k_m_per_warp == 0)
      << "M must be divisible by " << k_m_per_warp << ", but got " << M;
  ICHECK(N % k_n_per_warp == 0)
      << "N must be divisible by " << k_n_per_warp << ", but got " << N;

  if (policy.IsFullRow()) {
    m_warp = num_warps;
    n_warp = 1;
    if (M % (m_warp * k_m_per_warp) != 0) {
      int max_m_warps = M / k_m_per_warp;
      m_warp = max_m_warps;
      n_warp = num_warps / m_warp;
      if (n_warp == 0)
        n_warp = 1;
    }
  } else if (policy.IsFullCol()) {
    m_warp = 1;
    n_warp = num_warps;
    if (N % (n_warp * k_n_per_warp) != 0) {
      int max_n_warps = N / k_n_per_warp;
      n_warp = max_n_warps;
      m_warp = num_warps / n_warp;
      if (m_warp == 0)
        m_warp = 1;
    }
  } else if (policy.IsSquare()) {
    int max_m_warps = M / k_m_per_warp;
    float ideal_ratio = N > 0 ? static_cast<float>(M) / N : 1.0f;

    int best_m = 1;
    int best_n = 1;
    float best_balance = std::numeric_limits<float>::max();
    for (int m = 1; m <= max_m_warps && m <= num_warps; m++) {
      int n = num_warps / m;

      float m_per_warp = static_cast<float>(M) / (m * k_m_per_warp);
      float n_per_warp = static_cast<float>(N) / (n * k_n_per_warp);
      if (m_per_warp < 1 || n_per_warp < 1)
        continue;
      if (m * n != num_warps)
        continue;

      float balance = std::abs(m_per_warp / n_per_warp - ideal_ratio);
      if (balance < best_balance) {
        best_balance = balance;
        best_m = m;
        best_n = n;
      }
    }

    m_warp = best_m;
    n_warp = best_n;
  } else {
    ICHECK(0) << "Unknown GemmSPWarpPolicy";
  }

  ICHECK(m_warp * n_warp == num_warps)
      << "m_warp * n_warp must equal num_warps, m_warp: " << m_warp
      << ", n_warp: " << n_warp << ", num_warps: " << num_warps;
  policy.m_warp = m_warp;
  policy.n_warp = n_warp;
  return {m_warp, n_warp};
}

} // namespace

struct GemmSP {
  static String SelectInst(const GemmSPNode &op, int block_size,
                           Target target) {
    if (op.M % 16 == 0 && op.N % 16 == 0) {
      return kMacaMMASP;
    }
    ICHECK(op.M % 8 == 0 && op.N % 8 == 0)
        << "MACA gemm_sp requires M and N divisible by 8, got M=" << op.M
        << " N=" << op.N;
    return kMacaMMASPM8N8;
  }

  static std::pair<int, int>
  ComputeWarpPartition(const GemmSPWarpPolicyNode &policy, int M, int N,
                       int block_size, Target target, String gemm_inst) {
    int num_warps = block_size / TargetMacaGetWarpSize(target);
    if (gemm_inst == kMacaMMASPM8N8) {
      return ComputeDefaultWarpPartition(policy, M, N, num_warps, 8, 8);
    }
    return ComputeDefaultWarpPartition(policy, M, N, num_warps, 16, 16);
  }

  static bool ReuseExistingSharedLayout(String gemm_inst) {
    return gemm_inst == kMacaMMASP || gemm_inst == kMacaMMASPM8N8;
  }

  static String InstructionKind(String gemm_inst) {
    if (gemm_inst == kMacaMMASP || gemm_inst == kMacaMMASPM8N8) {
      return "mma.sp";
    }
    return "unknown";
  }
};

} // namespace maca

namespace {

bool MatchMacaGemmSPTarget(Target target) {
  return TargetIsMaca(target) || TargetIsCuTeDSL(target);
}

bool RegisterMacaGemmSP() {
  RegisterGemmSPImpl(GemmSPImpl{
      "maca.GemmSP",
      MatchMacaGemmSPTarget,
      maca::GemmSP::SelectInst,
      maca::GemmSP::ComputeWarpPartition,
      maca::GemmSP::ReuseExistingSharedLayout,
  });
  return true;
}

const bool maca_gemm_registered = RegisterMacaGemmSP();

} // namespace

} // namespace tl
} // namespace tvm
