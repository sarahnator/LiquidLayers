#pragma once

// =============================================================================
//  mpm_fluid.h  (v2)
//
//  Changes from v1
//  ───────────────
//  1. J stabilisation
//     The explicit Euler update  J *= (1 + dt·tr(C))  is replaced by the
//     multiplicative exponential update  J *= exp(dt·tr(C)).  This is the
//     exact solution to  dJ/dt = J·div(v)  under a frozen velocity gradient,
//     and it is unconditionally positive — J can never go negative or zero
//     regardless of how large dt·tr(C) is.  We still clamp J ∈ [J_min, J_max]
//     as a secondary guard, but the blow-up that happened on impact (where
//     dt·tr(C) reached -1.2e7) can no longer produce a negative J.
//
//     Additionally, kirchhoffStress() applies a symmetric pressure cap:
//       p = clamp(p, -pCap, +pCap)
//     The cap prevents the Tait EOS from generating arbitrarily large pressures
//     when J is near J_min, which is what was launching particles across the
//     domain after the J clamp fired.
//
//  2. RT band ordering fix
//     initialize() now accepts a BlockSpec struct that includes an
//     invert_bands flag.  When true, the top and bottom fluid assignments are
//     swapped, so the denser fluid starts on top (gravitationally unstable,
//     producing Rayleigh-Taylor fingering).
//
//  3. Dynamic fluid registry
//     FluidID is a uint8_t index into a runtime std::vector<FluidParams>.
//     The simulator supports up to kMaxFluids = 8 simultaneous types.
//     addFluid() / removeFluid() let the UI register and deregister fluids.
//     addBlock() seeds a new rectangular region of particles without
//     clearing existing ones, so the user can drop additional blocks at will.
//
//  Constitutive model (unchanged from v1)
//  ───────────────────────────────────────
//      τ = -J·p·I  +  J·μ·(C + Cᵀ)
//      p = K · (J^{-γ} - 1)        [Tait EOS]
// =============================================================================

#include <Eigen/Dense>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

// ── Constants
// ─────────────────────────────────────────────────────────────────
static constexpr int kMaxFluids = 8;
static constexpr float kJ_min = 0.2f;       // hard lower bound on Jacobian
static constexpr float kJ_max = 5.0f;       // hard upper bound on Jacobian
static constexpr float kPressureCap = 5e4f; // maximum |pressure| in stress [Pa]

// ── FluidParams
// ───────────────────────────────────────────────────────────────
struct FluidParams {
  std::string name = "Fluid";
  float density0 = 1000.f;    // rest density ρ₀    [kg/m²]
  float bulk_modulus = 200.f; // Tait stiffness K   [Pa]
  float gamma = 4.f;          // Tait exponent      [-]
  float viscosity = 0.01f;    // dynamic viscosity μ [Pa·s]
  std::array<float, 3> color = {0.5f, 0.5f, 0.5f};

  // Wave speed c₀ = sqrt(K·γ/ρ₀) — used for CFL timestep estimate.
  float c0 = 0.f;
  void computeDerived() {
    c0 = std::sqrt(bulk_modulus * gamma / std::max(density0, 1.f));
  }
};

// ── FluidID
// ─────────────────────────────────────────────────────────────────── Index
// into the simulator's runtime fluid registry.  uint8_t keeps the per-particle
// memory footprint small.
using FluidID = uint8_t;

// ── SimParams
// ─────────────────────────────────────────────────────────────────
struct SimParams {
  float domain_w = 10.f;
  float domain_h = 6.f;
  int grid_nx = 80;
  int grid_ny = 48;
  int ppc = 4;     // particle seeding: ppc×ppc per grid cell
  float dt = 2e-3; // 1e-4f;
  float gravity = -9.8f;

  // Derived
  float dx = 1.f;    // cell size = domain_w / grid_nx
  float D_inv = 4.f; // APIC D⁻¹ for quadratic B-splines = 4/dx²

  void computeDerived() {
    dx = domain_w / static_cast<float>(grid_nx);
    D_inv = 4.f / (dx * dx);
  }

  // CFL-safe dt.  Safety factor 0.3; raise to ~0.4 if bulk moduli are low.
  float estimateDt(const std::vector<FluidParams> &fluids) const {
    float c_max = 0.f;
    for (const auto &f : fluids)
      c_max = std::max(c_max, f.c0);
    return (c_max > 1e-6f) ? 0.3f * dx / c_max : dt;
  }
};

// ── Particle
// ──────────────────────────────────────────────────────────────────
struct Particle {
  Eigen::Vector2f pos = Eigen::Vector2f::Zero();
  Eigen::Vector2f vel = Eigen::Vector2f::Zero();
  Eigen::Matrix2f C = Eigen::Matrix2f::Zero(); // APIC affine matrix
  float J = 1.f;    // volume ratio det(F); guaranteed > 0
  float mass = 1.f; // [kg]
  float vol0 = 1.f; // reference volume [m²], refreshed each step
  FluidID fluid = 0;
};

// ── GridNode
// ──────────────────────────────────────────────────────────────────
struct GridNode {
  float mass = 0.f;
  Eigen::Vector2f momentum = Eigen::Vector2f::Zero();
  Eigen::Vector2f vel = Eigen::Vector2f::Zero();
  Eigen::Vector2f vel_new = Eigen::Vector2f::Zero();
  Eigen::Vector2f force = Eigen::Vector2f::Zero();
};

// ── WeightStencil
// ─────────────────────────────────────────────────────────────
//  Quadratic B-spline weights and gradients for a 3×3 node stencil.
//
//  Given particle position x_p and cell size dx:
//    base = floor(x_p/dx - 0.5)  clamped to [0, n-3]
//    ξ    = x_p/dx - base          (falls in roughly [0.5, 1.5))
//
//  1-D weights:
//    w₀ = ½(1.5 - ξ)²
//    w₁ = ¾ - (ξ-1)²
//    w₂ = ½(ξ - 0.5)²
//
//  1-D weight derivatives w.r.t. x_p (chain rule through ξ):
//    dw₀/dx_p = -(1.5-ξ)/dx
//    dw₁/dx_p = -2(ξ-1)/dx
//    dw₂/dx_p =  (ξ-0.5)/dx
//
//  2-D: w_{ab} = wx[a]·wy[b],  ∇w_{ab} = (dwx[a]·wy[b], wx[a]·dwy[b])
struct WeightStencil {
  int base_i, base_j;
  float wx[3], wy[3];
  float dwx[3], dwy[3];

  WeightStencil(const Eigen::Vector2f &xp, float dx, int nx, int ny) {
    auto init1D = [&](float coord, int n, int &base, float w[3], float dw[3]) {
      float s = coord / dx;
      base = std::max(0, std::min((int)(s - 0.5f), n - 3));
      float xi = s - (float)base;
      float d0 = 1.5f - xi, d1 = xi - 1.f, d2 = xi - 0.5f;
      w[0] = 0.5f * d0 * d0;
      w[1] = 0.75f - d1 * d1;
      w[2] = 0.5f * d2 * d2;
      dw[0] = -d0 / dx;
      dw[1] = -2.f * d1 / dx;
      dw[2] = d2 / dx;
    };
    init1D(xp.x(), nx, base_i, wx, dwx);
    init1D(xp.y(), ny, base_j, wy, dwy);
  }

  float weight(int a, int b) const { return wx[a] * wy[b]; }
  Eigen::Vector2f weightGrad(int a, int b) const {
    return {dwx[a] * wy[b], wx[a] * dwy[b]};
  }
};

// ── BlockSpec
// ─────────────────────────────────────────────────────────────────
//  Describes a rectangular particle block to seed (used by initialize and
//  addBlock).  The block is split into two horizontal bands separated by an
//  optional gap.
struct BlockSpec {
  float x_min = 3.f, x_max = 7.f;
  float y_min = 1.f, y_max = 5.f;
  float layer_gap = 0.02f;
  FluidID bottom_fluid = 0; // fluid in the lower half
  FluidID top_fluid = 1;    // fluid in the upper half
  // When true: the bottom_fluid is placed on TOP and top_fluid on BOTTOM.
  // Use this to create a Rayleigh-Taylor unstable configuration where the
  // denser fluid starts above the lighter one.
  bool invert_bands = false;
};

// ── FluidSimulation
// ────────────────────────────────────────────────────────
class FluidSimulation {
public:
  explicit FluidSimulation(SimParams params = {});

  // ── Fluid registry ────────────────────────────────────────────────────────
  int numFluids() const { return (int)fluids_.size(); }

  // Register a new fluid.  Returns the FluidID assigned (0-based index).
  // Returns -1 if kMaxFluids would be exceeded.
  int addFluid(FluidParams fp);

  // Remove fluid at index id.  Returns false if id is out of range or it
  // would leave the registry empty.  Existing particles keep their ID but
  // their params become undefined until the slot is refilled or they are
  // cleared.
  bool removeFluid(FluidID id);

  // How many live particles carry a given FluidID.
  int particleCountForFluid(FluidID id) const;

  const FluidParams &fluidParams(FluidID id) const { return fluids_.at(id); }
  FluidParams &fluidParamsMutable(FluidID id) { return fluids_.at(id); }
  const std::vector<FluidParams> &allFluids() const { return fluids_; }

  // ── Simulation lifetime ───────────────────────────────────────────────────
  // Clear all particles and seed from block.
  void initialize(const BlockSpec &block);

  // Add particles from block without clearing existing ones.
  void addBlock(const BlockSpec &block);

  void clearParticles() { particles_.clear(); }
  void step();

  // ── Accessors ─────────────────────────────────────────────────────────────
  const std::vector<Particle> &particles() const { return particles_; }
  const SimParams &params() const { return params_; }
  SimParams &paramsMutable() { return params_; }
  int frame() const { return frame_; }

private:
  void clearGrid();
  void P2G_mass();
  void volumeRecompute();
  void P2G_stress();
  void gridUpdate();
  void G2P_advect();

  Eigen::Matrix2f kirchhoffStress(const Particle &p) const;
  void seedBand(float x0, float x1, float y0, float y1, FluidID fid);

  int gridIdx(int i, int j) const { return j * params_.grid_nx + i; }
  bool inGrid(int i, int j) const {
    return i >= 0 && i < params_.grid_nx && j >= 0 && j < params_.grid_ny;
  }

  SimParams params_;
  std::vector<FluidParams> fluids_;
  std::vector<Particle> particles_;
  std::vector<GridNode> grid_;
  int frame_ = 0;
};
