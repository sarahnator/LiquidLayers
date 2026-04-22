#pragma once

// =============================================================================
//  mpm_fluid.h
// =============================================================================
//  This file defines a small 2D weakly-compressible MPM/APIC liquid solver with
//  a runtime fluid registry.
//
//  What the current code actually supports:
//    1. Multiple runtime-configurable fluids
//       - FluidID is a uint8_t index into std::vector<FluidParams>.
//       - Up to kMaxFluids fluid types can coexist.
//       - addFluid()/removeFluid() edit that registry at runtime.
//
//    2. Rectangular two-band block seeding
//       - initialize(block) clears all particles and seeds one block.
//       - addBlock(block) appends another block without clearing the old ones.
//       - A BlockSpec always creates two horizontal bands:
//           lower half -> bottom_fluid
//           upper half -> top_fluid
//       - There is no longer an invert_bands flag in the live code. If you want
//         a Rayleigh-Taylor-unstable configuration, assign the denser material
//         to top_fluid yourself.
//
//    3. Mouse-driven interaction modes applied on the grid
//       - GRAB: radial pull toward the cursor.
//       - WHIRL_POOL: tangential spin plus inward pull, implemented as a target
//         velocity field blended into grid velocities.
//       - A legacy tangential-only whirl path exists in comments in the .cpp
//         file but is intentionally disabled.
//
//  Constitutive model used by the current solver:
//    τ = -J p I + J μ (C + Cᵀ)
//    p = K (J^{-γ} - 1)          [Tait EOS]
//
//  where:
//    J  = local volume ratio proxy
//    C  = APIC affine velocity matrix / local velocity-gradient estimate
//    μ  = viscosity
//    K, γ = Tait EOS parameters
// =============================================================================

#include <Eigen/Dense>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

// ── Constants ────────────────────────────────────────────────────────────────
static constexpr int kMaxFluids = 8;
static constexpr float kJ_min = 0.2f;       // lower safety clamp on J
static constexpr float kJ_max = 5.0f;       // upper safety clamp on J
static constexpr float kPressureCap = 5e4f; // optional symmetric pressure cap

// ── FluidParams ──────────────────────────────────────────────────────────────
// Per-material parameters for the weakly-compressible liquid model.
struct FluidParams {
  std::string name = "Fluid";
  float density0 = 1000.f;    // reference density ρ₀ [kg/m² in this 2D setup]
  float bulk_modulus = 200.f; // Tait stiffness K [Pa]
  float gamma = 4.f;          // Tait exponent γ [-]
  float viscosity = 0.01f;    // dynamic viscosity μ [Pa·s]
  std::array<float, 3> color = {0.5f, 0.5f, 0.5f};

  // Small-signal wave speed estimate used for a CFL-style dt suggestion.
  float c0 = 0.f;
  void computeDerived() {
    c0 = std::sqrt(bulk_modulus * gamma / std::max(density0, 1.f));
  }
};

// ── FluidID ──────────────────────────────────────────────────────────────────
// Index into the runtime fluid registry.
using FluidID = uint8_t;

// ── SimParams ────────────────────────────────────────────────────────────────
struct SimParams {
  float domain_w = 10.f;
  float domain_h = 6.f;
  int grid_nx = 80;
  int grid_ny = 48;
  int ppc = 4;     // seed ppc×ppc particles per grid cell area
  float dt = 2e-3; // simulation timestep
  float gravity = -9.8f;

  // Derived quantities updated by computeDerived().
  float dx = 1.f;    // grid spacing = domain_w / grid_nx
  float D_inv = 4.f; // APIC normalization for quadratic B-splines = 4/dx²

  void computeDerived() {
    dx = domain_w / static_cast<float>(grid_nx);
    D_inv = 4.f / (dx * dx);
  }

  // CFL-style dt estimate using the largest wave speed among registered fluids.
  float estimateDt(const std::vector<FluidParams> &fluids) const {
    float c_max = 0.f;
    for (const auto &f : fluids)
      c_max = std::max(c_max, f.c0);
    return (c_max > 1e-6f) ? 0.3f * dx / c_max : dt;
  }
};

// ── Particle ─────────────────────────────────────────────────────────────────
struct Particle {
  Eigen::Vector2f pos = Eigen::Vector2f::Zero();
  Eigen::Vector2f vel = Eigen::Vector2f::Zero();
  Eigen::Matrix2f C = Eigen::Matrix2f::Zero(); // APIC affine velocity matrix
  float J = 1.f;    // scalar compression / expansion proxy
  float mass = 1.f; // particle mass
  float vol0 = 1.f; // effective particle volume used in stress scatter
  FluidID fluid = 0;
};

// ── GridNode ─────────────────────────────────────────────────────────────────
struct GridNode {
  float mass = 0.f;
  Eigen::Vector2f momentum = Eigen::Vector2f::Zero();
  Eigen::Vector2f vel = Eigen::Vector2f::Zero();
  Eigen::Vector2f vel_new = Eigen::Vector2f::Zero();
  Eigen::Vector2f force = Eigen::Vector2f::Zero();
};

// ── WeightStencil ────────────────────────────────────────────────────────────
// Quadratic B-spline weights and gradients over a 3×3 node stencil.
//
// For particle position x_p and grid spacing dx:
//   base = floor(x_p/dx - 0.5), clamped so the 3×3 stencil stays in-bounds
//   ξ    = x_p/dx - base
//
// The code stores both the weights and their spatial derivatives so the same
// helper can be reused for:
//   - P2G mass / momentum transfer
//   - density / volume recomputation
//   - stress force scatter
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

// ── BlockSpec ────────────────────────────────────────────────────────────────
// Description of a rectangular block that is split into two horizontal layers.
//
// The seeded geometry is:
//   lower band : [y_min,                y_min + layer_h]
//   upper band : [y_min + layer_h + gap, y_max]
//
// with layer_h = 0.5 * (y_max - y_min - layer_gap).
struct BlockSpec {
  float x_min = 3.f, x_max = 7.f;
  float y_min = 1.f, y_max = 5.f;
  float layer_gap = 0.02f;
  FluidID bottom_fluid = 0;
  FluidID top_fluid = 1;
};

// ── MouseForceMode ───────────────────────────────────────────────────────────
enum class MouseForceMode { GRAB, WHIRL_LEGACY, WHIRL_POOL };

// ── FluidSimulation ──────────────────────────────────────────────────────────
class FluidSimulation {
public:
  explicit FluidSimulation(SimParams params = {});

  // Fluid registry ------------------------------------------------------------
  int numFluids() const { return (int)fluids_.size(); }

  // Register a new fluid. Returns its FluidID, or -1 if the registry is full.
  int addFluid(FluidParams fp);

  // Remove a fluid slot. At least one fluid must remain.
  // The implementation compacts the registry and shifts particle FluidIDs above
  // the removed slot down by one.
  bool removeFluid(FluidID id);

  int particleCountForFluid(FluidID id) const;

  const FluidParams &fluidParams(FluidID id) const { return fluids_.at(id); }
  FluidParams &fluidParamsMutable(FluidID id) { return fluids_.at(id); }
  const std::vector<FluidParams> &allFluids() const { return fluids_; }

  // Simulation lifetime -------------------------------------------------------
  void initialize(const BlockSpec &block); // clear all particles, then seed

  // Mouse-driven forcing ------------------------------------------------------
  // These forces are injected on the grid after gridUpdate() and before G2P.
  void setMouseForce(Eigen::Vector2f pos, float radius, float strength,
                     MouseForceMode mode, bool active);
  void clearMouseForce() { mouse_active_ = false; }

  // Add another two-band block without clearing existing particles.
  void addBlock(const BlockSpec &block);

  void clearParticles() { particles_.clear(); }
  void step();

  // Accessors ----------------------------------------------------------------
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
  void applyMouseForcesToGrid(float sub_dt);
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

  // Mouse force state ---------------------------------------------------------
  bool mouse_active_ = false;
  Eigen::Vector2f mouse_pos_ = Eigen::Vector2f::Zero();
  float mouse_radius_ = 0.8f;
  float mouse_strength_ = 5.f;
  MouseForceMode mouse_mode_ = MouseForceMode::GRAB;
};
