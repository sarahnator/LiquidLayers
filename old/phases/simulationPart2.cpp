#include "simulation.h"
#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include <algorithm>
#include <cmath>
#include <iostream>

// ─────────────────────────────────────────────────────────────────────────────
Simulation::Simulation(SimParams params) : params_(std::move(params)) {
  params_.computeDerived();
  grid_.resize(params_.grid_nx * params_.grid_ny);
}

// ─────────────────────────────────────────────────────────────────────────────
//  initialize()
// ─────────────────────────────────────────────────────────────────────────────
void Simulation::initialize() {
  particles_.clear();
  frame_ = 0;

  const float W = params_.domain_w;
  const float H = params_.domain_h;
  const float pct = params_.layer_pct;
  const int ppc = params_.ppc;
  const float dx = params_.dx;

  struct Layer {
    MaterialType mat;
    float y_lo;
    float y_hi;
    float density;
  };
  std::vector<Layer> layers = {
      {MaterialType::Rock, 0.0f * H * pct, 1.0f * H * pct, 2500.f},
      {MaterialType::Sand, 1.0f * H * pct, 2.0f * H * pct, 1600.f},
      {MaterialType::Soil, 2.0f * H * pct, 3.0f * H * pct, 1300.f},
      {MaterialType::Water, 3.0f * H * pct, 4.0f * H * pct, 1000.f},
  };

  // Particle spacing: subdivide each grid cell into ppc x ppc
  const float px = dx / static_cast<float>(ppc);
  const float py = dx / static_cast<float>(ppc);

  auto jitter = [&](int i, float scale) -> float {
    return scale *
           (static_cast<float>((i * 1013904223 + 1664525) & 0xFFFF) / 65535.f -
            0.5f);
  };

  int pidx = 0;
  for (const auto &layer : layers) {
    for (float y = layer.y_lo + py * 0.5f; y < layer.y_hi; y += py) {
      for (float x = px * 0.5f; x < W; x += px) {
        Particle p;
        p.pos.x() = x + jitter(pidx * 2, px * 0.3f);
        p.pos.y() = y + jitter(pidx * 2 + 1, py * 0.3f);
        p.pos.x() = std::clamp(p.pos.x(), 0.001f, W - 0.001f);
        p.pos.y() = std::clamp(p.pos.y(), 0.001f, H - 0.001f);
        p.material = layer.mat;
        p.vel = Eigen::Vector2f::Zero();
        p.C = Eigen::Matrix2f::Zero();
        p.F = Eigen::Matrix2f::Identity();
        p.density0 = layer.density;

        // Particle mass: density × volume
        // Each particle represents px*py area (2D "volume")
        p.mass = layer.density * px * py;
        p.vol0 = px * py;

        particles_.push_back(p);
        ++pidx;
      }
    }
  }

  std::cout << "[MPM] Initialized " << particles_.size() << " particles, "
            << "grid " << params_.grid_nx << "x" << params_.grid_ny
            << " dx=" << params_.dx << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  clearGrid()  — zero all grid nodes at the start of each step
// ─────────────────────────────────────────────────────────────────────────────
void Simulation::clearGrid() {
  for (auto &node : grid_) {
    node.mass = 0.f;
    node.momentum = Eigen::Vector2f::Zero();
    node.vel = Eigen::Vector2f::Zero();
    node.vel_new = Eigen::Vector2f::Zero();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  substep_P2G()
//
//  For each particle p, scatter its mass and momentum to the 3×3 grid stencil.
//
//  Mass transfer:
//    m_i += w_ip * m_p
//
//  Momentum transfer (APIC):
//    mv_i += w_ip * m_p * (v_p + C_p * (x_i - x_p))
//
//  The APIC term  C_p * (x_i - x_p)  adds the local velocity gradient
//  contribution.  Without it this degenerates to plain PIC, which is
//  overly diffusive.
// ─────────────────────────────────────────────────────────────────────────────
void Simulation::substep_P2G() {
  const float dx = params_.dx;
  const int nx = params_.grid_nx;
  const int ny = params_.grid_ny;

  for (const auto &p : particles_) {
    WeightStencil ws(p.pos, dx, nx, ny);

    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        int ni = ws.base_i + a;
        int nj = ws.base_j + b;
        if (!inGrid(ni, nj))
          continue;

        float w = ws.weight(a, b);

        // Vector from particle to this grid node (world units)
        Eigen::Vector2f xip = Eigen::Vector2f(static_cast<float>(ni) * dx,
                                              static_cast<float>(nj) * dx) -
                              p.pos;

        // APIC momentum contribution: v_p + C_p * (x_i - x_p)
        Eigen::Vector2f mv = p.mass * w * (p.vel + p.C * xip);

        auto &node = grid_[gridIdx(ni, nj)];
        node.mass += w * p.mass;
        node.momentum += mv;
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  substep_gridUpdate()
//
//  For each grid node:
//    1. Derive velocity from momentum:  v = mv / m   (if m > threshold)
//    2. Apply external forces (gravity here; stress in Phase 3)
//    3. Enforce boundary conditions (sticky walls)
//
//  This is the grid-space analogue of F=ma from your mass-spring sim —
//  except instead of per-particle forces, we operate on grid nodes.
// ─────────────────────────────────────────────────────────────────────────────
void Simulation::substep_gridUpdate() {
  const float dt = params_.dt;
  const float g = params_.gravity;
  const int nx = params_.grid_nx;
  const int ny = params_.grid_ny;
  const float dx = params_.dx;

  // Boundary thickness (nodes from edge that are "wall" nodes)
  const int wall = 2;

  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      auto &node = grid_[gridIdx(i, j)];

      if (node.mass < 1e-10f)
        continue; // empty node — skip

      // Recover velocity from momentum
      node.vel = node.momentum / node.mass;

      // Apply gravity
      node.vel_new = node.vel;
      node.vel_new.y() += g * dt;

      // ── Boundary conditions  (sticky walls on all 4 sides) ───────────
      // A sticky (no-slip) wall zeroes the velocity component
      // pointing into the wall.  A free-slip wall would only zero
      // the normal component — you can experiment with that later.
      bool on_left = (i < wall);
      bool on_right = (i >= nx - wall);
      bool on_bottom = (j < wall);
      bool on_top = (j >= ny - wall);

      if (on_left && node.vel_new.x() < 0.f)
        node.vel_new.x() = 0.f;
      if (on_right && node.vel_new.x() > 0.f)
        node.vel_new.x() = 0.f;
      if (on_bottom && node.vel_new.y() < 0.f)
        node.vel_new.y() = 0.f;
      if (on_top && node.vel_new.y() > 0.f)
        node.vel_new.y() = 0.f;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  substep_G2P()
//
//  For each particle p, gather updated velocity from its 3×3 grid stencil.
//
//  Velocity update (APIC):
//    v_p = sum_i  w_ip * v_i_new
//
//  APIC affine matrix update:
//    C_p = D_inv * sum_i  w_ip * v_i_new * (x_i - x_p)^T
//
//  D_inv = 4/dx²  for quadratic B-spline.
//  C carries the local velocity gradient, so the next P2G step can
//  reconstruct a smooth velocity field without numerical dissipation.
//
//  Deformation gradient update (needed for constitutive model in Phase 3):
//    F_p_new = (I + dt * grad_v_p) * F_p
//  where  grad_v_p ≈ C_p  (APIC gives us grad_v for free)
// ─────────────────────────────────────────────────────────────────────────────
void Simulation::substep_G2P() {
  const float dx = params_.dx;
  const float D_inv = params_.D_inv;
  const float dt = params_.dt;
  const int nx = params_.grid_nx;
  const int ny = params_.grid_ny;

  for (auto &p : particles_) {
    WeightStencil ws(p.pos, dx, nx, ny);

    Eigen::Vector2f v_new = Eigen::Vector2f::Zero();
    Eigen::Matrix2f C_new = Eigen::Matrix2f::Zero();

    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        int ni = ws.base_i + a;
        int nj = ws.base_j + b;
        if (!inGrid(ni, nj))
          continue;

        float w = ws.weight(a, b);
        const Eigen::Vector2f &vi = grid_[gridIdx(ni, nj)].vel_new;

        // Vector from particle to grid node
        Eigen::Vector2f xip = Eigen::Vector2f(static_cast<float>(ni) * dx,
                                              static_cast<float>(nj) * dx) -
                              p.pos;

        // Accumulate velocity (standard PIC part)
        v_new += w * vi;

        // Accumulate APIC affine matrix
        // C += w * v_i ⊗ (x_i - x_p)   (outer product)
        C_new += w * (vi * xip.transpose());
      }
    }

    p.vel = v_new;
    p.C = D_inv * C_new;

    // Update deformation gradient:  F_new = (I + dt*C) * F
    // C ≈ grad_v, so (I + dt*C) is the incremental deformation.
    // In Phase 3 this will drive the constitutive model.
    Eigen::Matrix2f F_inc = Eigen::Matrix2f::Identity() + dt * p.C;
    p.F = F_inc * p.F;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  substep_advect()
//
//  Move particles forward by their updated velocity.
//  Clamp positions to stay inside the domain.
//  (In Phase 3 you can switch to a more sophisticated scheme if needed.)
// ─────────────────────────────────────────────────────────────────────────────
void Simulation::substep_advect() {
  const float dt = params_.dt;
  const float W = params_.domain_w;
  const float H = params_.domain_h;

  for (auto &p : particles_) {
    p.pos += dt * p.vel;
    p.pos.x() = std::clamp(p.pos.x(), 0.001f, W - 0.001f);
    p.pos.y() = std::clamp(p.pos.y(), 0.001f, H - 0.001f);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  step()  — one full MPM timestep
//
//  The canonical MPM loop:
//    1. Clear grid
//    2. P2G  (particles paint mass + momentum onto grid)
//    3. Grid update  (forces + gravity + boundaries)
//    4. G2P  (grid paints velocity back to particles, updates C and F)
//    5. Advect  (move particles with new velocity)
//
//  Without a constitutive model (Phase 3), particles behave like a
//  pressureless gas — they'll fall under gravity and pile up at the
//  bottom, but won't exert pressure or resist compression.
//  That's expected and correct for Phase 2.
// ─────────────────────────────────────────────────────────────────────────────
void Simulation::step() {
  clearGrid();
  substep_P2G();
  substep_gridUpdate();
  substep_G2P();
  substep_advect();
  ++frame_;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Polyscope interface  (unchanged from Phase 1)
// ─────────────────────────────────────────────────────────────────────────────
void Simulation::buildRenderArrays(
    std::vector<std::array<double, 3>> &pos3d,
    std::vector<std::array<double, 3>> &colors) const {
  const size_t N = particles_.size();
  pos3d.resize(N);
  colors.resize(N);
  for (size_t i = 0; i < N; ++i) {
    const auto &p = particles_[i];
    pos3d[i] = {static_cast<double>(p.pos.x()), static_cast<double>(p.pos.y()),
                0.0};
    auto c = materialColor(p.material);
    colors[i] = {c[0], c[1], c[2]};
  }
}

void Simulation::registerPolyscope() {
  std::vector<std::array<double, 3>> pos3d, colors;
  buildRenderArrays(pos3d, colors);
  auto *cloud = polyscope::registerPointCloud(kCloudName, pos3d);
  cloud->setPointRadius(0.003);
  cloud->setPointRenderMode(polyscope::PointRenderMode::Sphere);
  cloud->addColorQuantity("material_color", colors)->setEnabled(true);
}

void Simulation::updatePolyscope() {
  std::vector<std::array<double, 3>> pos3d, colors;
  buildRenderArrays(pos3d, colors);
  auto *cloud = polyscope::getPointCloud(kCloudName);
  cloud->updatePointPositions(pos3d);
  cloud->addColorQuantity("material_color", colors)->setEnabled(true);
}
