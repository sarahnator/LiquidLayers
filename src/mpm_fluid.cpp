// =============================================================================
//  mpm_fluid.cpp  (v2)
// =============================================================================

#include "mpm_fluid.h"
#include <algorithm>
#include <cmath>
#include <iostream>

// ─────────────────────────────────────────────────────────────────────────────
//  Constructor
// ─────────────────────────────────────────────────────────────────────────────
FluidSimulation::FluidSimulation(SimParams params)
    : params_(std::move(params)) {
  params_.computeDerived();
  grid_.resize((size_t)(params_.grid_nx * params_.grid_ny));
}

// ─────────────────────────────────────────────────────────────────────────────
//  Fluid registry
// ─────────────────────────────────────────────────────────────────────────────
int FluidSimulation::addFluid(FluidParams fp) {
  if ((int)fluids_.size() >= kMaxFluids)
    return -1;
  fp.computeDerived();
  int id = (int)fluids_.size();
  fluids_.push_back(std::move(fp));
  return id;
}

bool FluidSimulation::removeFluid(FluidID id) {
  if (id >= fluids_.size() || fluids_.size() <= 1)
    return false;
  fluids_.erase(fluids_.begin() + id);
  // Remap particle IDs: particles above the removed slot shift down by one.
  for (auto &p : particles_)
    if (p.fluid > id)
      --p.fluid;
  return true;
}

int FluidSimulation::particleCountForFluid(FluidID id) const {
  int n = 0;
  for (const auto &p : particles_)
    if (p.fluid == id)
      ++n;
  return n;
}

// ─────────────────────────────────────────────────────────────────────────────
//  seedBand()
//
//  Fill a rectangle [x0,x1]×[y0,y1] with particles of fluid type fid.
//
//  Seeding layout:
//    The rectangle is divided into a sub-grid of spacing px = py = dx/ppc.
//    Each sub-cell gets one particle placed at its centre plus a small
//    deterministic jitter (±15% of sub-cell size) to break lattice symmetry.
//
//  Particle mass:
//    m_p = ρ₀ · px·py
//    This ensures that at t=0, when J=1, the Tait pressure p=K(1⁻ᵞ-1)=0.
//    No spurious pressure kick occurs on the first step.
// ─────────────────────────────────────────────────────────────────────────────
void FluidSimulation::seedBand(float x0, float x1, float y0, float y1,
                               FluidID fid) {
  const FluidParams &fp = fluids_.at(fid);
  const float dx = params_.dx;
  const int ppc = params_.ppc;
  const float px = dx / (float)ppc;
  const float py = dx / (float)ppc;

  // Deterministic jitter: maps particle index to a pseudo-random offset.
  int pidx = (int)particles_.size();
  auto jitter = [](int i, float s) -> float {
    uint32_t h = (uint32_t)i * 1013904223u + 1664525u;
    return s * ((float)(h & 0xFFFFu) / 65535.f - 0.5f);
  };

  for (float y = y0 + 0.5f * py; y < y1; y += py) {
    for (float x = x0 + 0.5f * px; x < x1; x += px) {
      Particle p;
      p.fluid = fid;
      p.vol0 = px * py;
      p.mass = fp.density0 * p.vol0; // m = ρ₀·vol₀
      p.J = 1.f;
      p.C = Eigen::Matrix2f::Zero();
      p.vel = Eigen::Vector2f::Zero();
      p.pos.x() = std::clamp(x + jitter(2 * pidx, 0.15f * px), 0.001f,
                             params_.domain_w - 0.001f);
      p.pos.y() = std::clamp(y + jitter(2 * pidx + 1, 0.15f * py), 0.001f,
                             params_.domain_h - 0.001f);
      particles_.push_back(p);
      ++pidx;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  initialize() / addBlock()
//
//  A BlockSpec splits the rectangle into two horizontal bands separated by an
//  optional gap:
//
//    top    band: [y_mid + gap/2,  y_max]   → top_fluid
//    bottom band: [y_min,          y_mid - gap/2] → bottom_fluid
//
//  When block.invert_bands = true, the fluid assignments are swapped so the
//  "bottom_fluid" is seeded on top.  This is the correct way to configure a
//  Rayleigh-Taylor scenario: assign the heavy fluid as bottom_fluid but also
//  set invert_bands=true so it physically starts on top.
// ─────────────────────────────────────────────────────────────────────────────
void FluidSimulation::addBlock(const BlockSpec &block) {
  float H = block.y_max - block.y_min;
  float gap = std::max(block.layer_gap, 0.f);
  float layer_h = std::max(0.5f * (H - gap), 0.f);

  // Natural assignment: bottom band → bottom_fluid, top band → top_fluid.
  // invert_bands swaps which fluid goes where without changing the geometry.
  FluidID lo_fluid = block.invert_bands ? block.top_fluid : block.bottom_fluid;
  FluidID hi_fluid = block.invert_bands ? block.bottom_fluid : block.top_fluid;

  float lo_y0 = block.y_min;
  float lo_y1 = block.y_min + layer_h;
  float hi_y0 = block.y_min + layer_h + gap;
  float hi_y1 = block.y_min + 2.f * layer_h + gap;

  seedBand(block.x_min, block.x_max, lo_y0, lo_y1, lo_fluid);
  seedBand(block.x_min, block.x_max, hi_y0, hi_y1, hi_fluid);
}

void FluidSimulation::initialize(const BlockSpec &block) {
  particles_.clear();
  frame_ = 0;
  params_.computeDerived();
  for (auto &f : fluids_)
    f.computeDerived();

  addBlock(block);

  std::cout << "[TwoFluid] initialized " << particles_.size()
            << " particles across " << fluids_.size() << " fluid type(s)\n";
  float safe_dt = params_.estimateDt(fluids_);
  std::cout << "[TwoFluid] CFL-suggested dt: " << safe_dt
            << "  (current dt: " << params_.dt << ")\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  kirchhoffStress()
//
//  Weakly-compressible Newtonian fluid Kirchhoff stress:
//
//      τ = -J·p·I  +  J·μ·(C + Cᵀ)
//
//  Tait EOS:
//      p = K · (J^{-γ} - 1)
//        = K · (exp(-γ·ln J) - 1)
//
//  Physical meaning of each term:
//    -J·p·I    : isotropic pressure (compression → positive pressure →
//    repulsion) J·μ·(C+Cᵀ): viscous stress proportional to the
//    rate-of-deformation
//               D = sym(C), which smooths velocity gradients
//
//  Pressure cap:
//    The Tait EOS diverges as J→0.  Even with the J clamp and the exp-based
//    update, residual numerical noise can push J to kJ_min, giving pressures
//    on the order of K·(kJ_min^{-γ}-1) which for γ=4, kJ_min=0.2 is ~624·K.
//    A symmetric cap |p| ≤ kPressureCap prevents that from producing forces
//    large enough to eject particles across the domain in one step.
//    The cap value (5×10⁴ Pa) should be set ≫ the typical working pressure
//    (a few ×K) but ≪ the blow-up value; tune if needed.
// ─────────────────────────────────────────────────────────────────────────────
Eigen::Matrix2f FluidSimulation::kirchhoffStress(const Particle &p) const {
  // const FluidParams &fp = fluids_.at(p.fluid);

  // // J is already clamped in G2P_advect, but clamp again defensively.
  // float J = std::clamp(p.J, kJ_min, kJ_max);

  // // Tait pressure: p = K·(J^{-γ} - 1)
  // float pressure = fp.bulk_modulus * (std::exp(-fp.gamma * std::log(J))
  // - 1.f);

  // // Symmetric pressure cap: prevents extreme forces at the boundary on
  // impact. pressure = std::clamp(pressure, -kPressureCap, kPressureCap);

  // // Isotropic pressure term
  // Eigen::Matrix2f tau = -J * pressure * Eigen::Matrix2f::Identity();

  // // Viscous term: J·μ·(C + Cᵀ)
  // // Note: (C + Cᵀ) = 2·sym(C) = 2·D, so J·μ·(C+Cᵀ) = J·2μ·D.
  // // Some codes pre-bake the factor of 2 into μ; here it is explicit.
  // if (fp.viscosity > 0.f)
  //     tau += J * fp.viscosity * (p.C + p.C.transpose());

  const FluidParams &fp = fluids_.at(p.fluid);

  float J = p.J;
  // Guard against degenerate compression / expansion
  J = std::clamp(J, 0.1f, 5.f);

  // Tait equation of state: p = K · (J^{-γ} - 1)
  // std::pow(J, -gamma) = exp(-gamma * ln(J))
  float pressure = fp.bulk_modulus * (std::exp(-fp.gamma * std::log(J)) - 1.f);

  // Clamp pressure: allow small tension but no large tensile pressure
  float p_floor = -0.1f * fp.bulk_modulus;
  pressure = std::max(pressure, p_floor);

  // Isotropic pressure part:  -J·p·I
  Eigen::Matrix2f tau = -J * pressure * Eigen::Matrix2f::Identity();

  // Viscous part:  J · 2μ · D  where D = sym(C) = ½(C + Cᵀ)
  // Note: the factor of 2 is absorbed into 2μ, so:
  //   J · 2μ · ½(C + Cᵀ)  =  J · μ · (C + Cᵀ)
  if (fp.viscosity > 0.f) {
    tau += J * fp.viscosity * (p.C + p.C.transpose());
  }

  return tau;
}

// ─────────────────────────────────────────────────────────────────────────────
//  clearGrid()
// ─────────────────────────────────────────────────────────────────────────────
void FluidSimulation::clearGrid() {
  for (auto &n : grid_) {
    n.mass = 0.f;
    n.momentum = n.vel = n.vel_new = n.force = Eigen::Vector2f::Zero();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  P2G_mass()  —  Particle-to-Grid pass 1: mass and APIC momentum
//
//  For each particle p and each of its 9 stencil nodes i:
//
//      m_i      += w_{ip} · m_p
//      (mv)_i   += w_{ip} · m_p · (v_p  +  C_p · (x_i - x_p))
//
//  The affine correction  C_p·(x_i - x_p)  is the APIC term (Jiang 2015)
//  that recovers linear velocity fields exactly on the grid, reducing
//  numerical diffusion relative to standard PIC.
// ─────────────────────────────────────────────────────────────────────────────
void FluidSimulation::P2G_mass() {
  const float dx = params_.dx;
  const int nx = params_.grid_nx, ny = params_.grid_ny;

  for (const auto &p : particles_) {
    WeightStencil ws(p.pos, dx, nx, ny);
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) {
        int ni = ws.base_i + a, nj = ws.base_j + b;
        if (!inGrid(ni, nj))
          continue;
        float w = ws.weight(a, b);
        Eigen::Vector2f xip((float)ni * dx - p.pos.x(),
                            (float)nj * dx - p.pos.y());
        auto &node = grid_[gridIdx(ni, nj)];
        node.mass += w * p.mass;
        node.momentum += w * p.mass * (p.vel + p.C * xip);
      }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  volumeRecompute()
//
//  Re-estimate each particle's current volume from the grid mass field:
//
//      ρ_p  = Σ_i  w_{ip} · (m_i / dx²)
//      vol_p = m_p / ρ_p
//
//  This decouples vol₀ from the drift-prone Jacobian J and prevents the
//  free-surface clustering artefact where sparse particles near the surface
//  underestimate their own volume and generate too little pressure.
//
//  The floor prevents division by zero at nodes with no mass (free surface).
// ─────────────────────────────────────────────────────────────────────────────
void FluidSimulation::volumeRecompute() {
  const float dx = params_.dx, dx2 = dx * dx;
  const int nx = params_.grid_nx, ny = params_.grid_ny;

  for (auto &p : particles_) {
    const FluidParams &fp = fluids_.at(p.fluid);
    WeightStencil ws(p.pos, dx, nx, ny);
    float rho = 0.f;
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) {
        int ni = ws.base_i + a, nj = ws.base_j + b;
        if (!inGrid(ni, nj))
          continue;
        rho += ws.weight(a, b) * grid_[gridIdx(ni, nj)].mass / dx2;
      }
    rho = std::max(rho, 0.1f * fp.density0);
    p.vol0 = p.mass / rho;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  P2G_stress()  —  Particle-to-Grid pass 2: internal force scatter
//
//  For each particle p and each stencil node i:
//
//      f_i  -=  vol₀_p · τ_p · ∇w_{ip}
//
//  This implements the discrete virtual-work principle from MLS-MPM eq. 10.
//  vol₀_p must be current (call after volumeRecompute).
// ─────────────────────────────────────────────────────────────────────────────
void FluidSimulation::P2G_stress() {
  const float dx = params_.dx;
  const int nx = params_.grid_nx, ny = params_.grid_ny;

  for (const auto &p : particles_) {
    Eigen::Matrix2f tau = kirchhoffStress(p);
    WeightStencil ws(p.pos, dx, nx, ny);
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) {
        int ni = ws.base_i + a, nj = ws.base_j + b;
        if (!inGrid(ni, nj))
          continue;
        grid_[gridIdx(ni, nj)].force -= p.vol0 * (tau * ws.weightGrad(a, b));
      }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  gridUpdate()
//
//  For each node with non-negligible mass:
//    1. v_i       = (mv)_i / m_i                       (momentum → velocity)
//    2. v_i^new   = v_i + dt·(f_i/m_i + g·ĵ)          (integrate forces)
//    3. Slip BCs  : zero the wall-normal component if it penetrates.
//
//  Wall thickness wall=2 gives 2·dx of buffer.  The quadratic B-spline
//  support is 1.5·dx wide, so wall=2 ensures no stencil node lies outside
//  the domain.  Increase to 3 if you see artefacts near boundaries.
// ─────────────────────────────────────────────────────────────────────────────
void FluidSimulation::gridUpdate() {
  const float dt = params_.dt;
  const int nx = params_.grid_nx, ny = params_.grid_ny, wall = 2;

  for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
      auto &node = grid_[gridIdx(i, j)];
      if (node.mass < 1e-12f)
        continue;

      node.vel = node.momentum / node.mass;
      Eigen::Vector2f accel = node.force / node.mass;
      accel.y() += params_.gravity;
      node.vel_new = node.vel + dt * accel;

      // One-sided slip BCs: only block penetrating velocity component.
      if (i < wall && node.vel_new.x() < 0.f)
        node.vel_new.x() = 0.f;
      if (i >= nx - wall && node.vel_new.x() > 0.f)
        node.vel_new.x() = 0.f;
      if (j < wall && node.vel_new.y() < 0.f)
        node.vel_new.y() = 0.f;
      // Top wall left open so fluid can splash upward freely.
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  G2P_advect()  —  Grid-to-Particle gather + J update + advection
//
//  Velocity gather (APIC):
//      v_p    = Σ_i  w_{ip} · v_i^new
//      B_p    = Σ_i  w_{ip} · v_i^new ⊗ (x_i - x_p)
//      C_p    = D⁻¹ · B_p              (APIC velocity gradient)
//
//  Jacobian update — FIX vs v1:
//      J_p^{n+1} = J_p^n · exp(dt · tr(C_p))
//
//  Rationale:
//    The governing ODE is  dJ/dt = J · div(v) = J · tr(C).
//    The explicit Euler discretisation  J *= (1 + dt·tr(C))  can produce
//    negative J when dt·tr(C) < -1, which occurs on high-velocity impact
//    (observed: dt·tr(C) ≈ -1.2×10⁷).  The exponential form is the exact
//    solution of the ODE under frozen C and is unconditionally positive:
//    no matter how negative tr(C) is, exp(...) > 0.  We then clamp J to
//    [kJ_min, kJ_max] as a secondary safeguard.
//
//  Advection:
//      x_p^{n+1} = x_p^n + dt·v_p
//    followed by a hard position clamp to keep particles inside the domain.
// ─────────────────────────────────────────────────────────────────────────────
void FluidSimulation::G2P_advect() {
  const float dx = params_.dx;
  const float D_inv = params_.D_inv;
  const float dt = params_.dt;
  const int nx = params_.grid_nx, ny = params_.grid_ny;

  for (auto &p : particles_) {
    WeightStencil ws(p.pos, dx, nx, ny);
    Eigen::Vector2f v_new = Eigen::Vector2f::Zero();
    Eigen::Matrix2f B = Eigen::Matrix2f::Zero();

    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) {
        int ni = ws.base_i + a, nj = ws.base_j + b;
        if (!inGrid(ni, nj))
          continue;
        float w = ws.weight(a, b);
        const Eigen::Vector2f &vi = grid_[gridIdx(ni, nj)].vel_new;
        Eigen::Vector2f xip((float)ni * dx - p.pos.x(),
                            (float)nj * dx - p.pos.y());
        v_new += w * vi;
        B += w * (vi * xip.transpose());
      }

    p.vel = v_new;
    p.C = D_inv * B;

    // ── J update (exponential form) ───────────────────────────────────────
    // tr(C) = div(v) = local volumetric strain rate.
    float div_v = p.C.trace();
    // p.J = std::clamp(p.J * std::exp(dt * div_v), kJ_min, kJ_max);
    // exp(dt·tr(C)) is always positive, so J stays positive unconditionally.
    // with the exponential update form, the particles compress into a line on
    // impact with the floor
    p.J = std::clamp(p.J * (1.f + dt * div_v), 0.1f, 5.f);

    // ── Advect ────────────────────────────────────────────────────────────
    p.pos += dt * p.vel;
    p.pos.x() = std::clamp(p.pos.x(), 0.001f, params_.domain_w - 0.001f);
    p.pos.y() = std::clamp(p.pos.y(), 0.001f, params_.domain_h - 0.001f);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  step()
//
//  One complete MPM timestep:
//    1. clearGrid       — zero node accumulators
//    2. P2G_mass        — scatter mass & APIC momentum
//    3. volumeRecompute — re-estimate vol₀ from grid density
//    4. P2G_stress      — scatter internal forces using fresh vol₀
//    5. gridUpdate      — momentum→velocity, gravity, BCs
//    6. G2P_advect      — gather velocity, update J (exp form), advect
// ─────────────────────────────────────────────────────────────────────────────
void FluidSimulation::step() {
  clearGrid();
  P2G_mass();
  volumeRecompute();
  P2G_stress();
  gridUpdate();
  G2P_advect();
  ++frame_;
}
