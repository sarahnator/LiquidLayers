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
    float y_lo, y_hi;
  };
  std::vector<Layer> layers = {
      {MaterialType::Rock, 0.f * H * pct, 1.f * H * pct},
      {MaterialType::Sand, 1.f * H * pct, 2.f * H * pct},
      {MaterialType::Soil, 2.f * H * pct, 3.f * H * pct},
      {MaterialType::Water, 3.f * H * pct, 4.f * H * pct},
  };

  const float px = dx / static_cast<float>(ppc);
  const float py = dx / static_cast<float>(ppc);

  auto jitter = [&](int i, float scale) -> float {
    return scale *
           (static_cast<float>((i * 1013904223 + 1664525) & 0xFFFF) / 65535.f -
            0.5f);
  };

  int pidx = 0;
  for (const auto &layer : layers) {
    MaterialParams mp = defaultMaterialParams(layer.mat);
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
        p.mass = mp.density0 * px * py;
        p.vol0 = px * py;
        particles_.push_back(p);
        ++pidx;
      }
    }
  }
  std::cout << "[MPM] " << particles_.size() << " particles  dx=" << dx << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  kirchhoffStress()
//
//  Returns tau_p = J_p * sigma_p  (Kirchhoff stress, 2x2 matrix).
//
//  The Kirchhoff stress is what appears in the grid force formula:
//    f_i = -sum_p  vol0_p * tau_p * grad_w_ip
//
//  For a weakly compressible fluid, sigma = -p*I + sigma_visc, so:
//    tau = J*sigma = -J*p*I + J*sigma_visc
//
//  Pressure from the Tait equation of state:
//    p_p = k * (J^{-gamma} - 1)
//
//  Viscous stress (symmetric velocity gradient):
//    sigma_visc = mu * (C + C^T)
//  We multiply by J to get the Kirchhoff viscous term.
//
//  Note: for the Kirchhoff stress we do NOT need F^{-T} explicitly because
//  for an isotropic pressure fluid, J*sigma*F^{-T}*F^T = J*sigma*I = J*sigma.
//  If you add anisotropic elasticity (Phase 4+) you will need the full formula.
// ─────────────────────────────────────────────────────────────────────────────
Eigen::Matrix2f Simulation::kirchhoffStress(const Particle &p) const {
  MaterialParams mp = defaultMaterialParams(p.material);

  // J = det(F) — ratio of current volume to reference volume
  float J = p.F.determinant();

  // Clamp J to a safe range to prevent NaN from extreme compression/expansion
  J = std::clamp(J, 0.1f, 10.0f);

  // ── Pressure (Tait equation of state) ─────────────────────────────────────
  // p = k * (J^{-gamma} - 1)
  // J^{-gamma} = exp(-gamma * log(J))  — numerically safer than std::pow
  float J_neg_gamma = std::exp(-mp.gamma * std::log(J));
  float pressure = mp.bulk_modulus * (J_neg_gamma - 1.f);

  // Kirchhoff pressure stress:  tau_pressure = -J * p * I
  // The minus sign: positive pressure resists compression (pushes outward).
  Eigen::Matrix2f tau = -J * pressure * Eigen::Matrix2f::Identity();

  // ── Viscous stress ─────────────────────────────────────────────────────────
  // sigma_visc = mu * (C + C^T)   where C ≈ grad_v (APIC affine matrix)
  // Kirchhoff viscous stress: tau_visc = J * mu * (C + C^T)
  // For small deformations J ≈ 1 and this reduces to the standard viscous term.
  if (mp.viscosity > 0.f) {
    Eigen::Matrix2f strain_rate = p.C + p.C.transpose();
    tau += J * mp.viscosity * strain_rate;
  }

  return tau;
}

// ─────────────────────────────────────────────────────────────────────────────
//  clearGrid()
// ─────────────────────────────────────────────────────────────────────────────
void Simulation::clearGrid() {
  for (auto &node : grid_) {
    node.mass = 0.f;
    node.momentum = Eigen::Vector2f::Zero();
    node.vel = Eigen::Vector2f::Zero();
    node.vel_new = Eigen::Vector2f::Zero();
    node.force = Eigen::Vector2f::Zero(); // Phase 3: clear forces too
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  substep_P2G()  — unchanged from Phase 2
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
        Eigen::Vector2f xip = Eigen::Vector2f(ni * dx, nj * dx) - p.pos;
        auto &node = grid_[gridIdx(ni, nj)];
        node.mass += w * p.mass;
        node.momentum += w * p.mass * (p.vel + p.C * xip);
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  substep_gridUpdate()
//
//  Phase 3 additions vs Phase 2:
//    1. A stress force scatter pass (particles → grid) using grad_w
//    2. node.force stores the total force (stress + gravity*mass)
//    3. Velocity update uses force/mass instead of just g*dt
//
//  The stress scatter:
//    For each particle p, compute tau_p = kirchhoffStress(p).
//    For each stencil node (i,j):
//        grad_w = weightGrad(a, b, dx)          (2-vector, units 1/m)
//        f_ip   = -vol0_p * tau_p * grad_w      (2-vector, units N)
//        node.force += f_ip
//
//  Why -vol0 and not -vol_current?
//    The formula f_i = -sum_p vol0_p * tau_p * grad_w comes from
//    differentiating the total energy E = sum_p vol0_p * Psi(F_p) with respect
//    to x_i. The Kirchhoff stress tau = J*sigma already accounts for the volume
//    change (the J factor converts reference to current volume), so we use vol0
//    here, not vol0*J.  Using vol0*J would double-count the volume change.
//
//  The velocity update:
//    v_new = v + dt * (f_stress + f_gravity) / mass
//          = v + dt * f_total / mass
//    where f_gravity = mass * g * e_y  (already divided by mass below)
// ─────────────────────────────────────────────────────────────────────────────
void Simulation::substep_gridUpdate() {
  const float dx = params_.dx;
  const int nx = params_.grid_nx;
  const int ny = params_.grid_ny;
  const int wall = 2;

  // ── Pass 1: scatter stress forces from particles to grid ──────────────────
  for (const auto &p : particles_) {
    Eigen::Matrix2f tau = kirchhoffStress(p);
    WeightStencil ws(p.pos, dx, nx, ny);

    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        int ni = ws.base_i + a;
        int nj = ws.base_j + b;
        if (!inGrid(ni, nj))
          continue;

        // grad_w: gradient of weight w.r.t. particle position x_p
        // Units: 1/m.  Derived via chain rule through B-spline (see types.h).
        Eigen::Vector2f grad_w = ws.weightGrad(a, b, dx);

        // Force contribution:  f_ip = -vol0_p * tau_p * grad_w_ip
        // tau is 2x2, grad_w is 2x1, result is 2x1 force vector.
        // The negative sign: stress resists deformation, so compression
        // (positive pressure) pushes nodes apart (negative force on inward
        // nodes).
        grid_[gridIdx(ni, nj)].force -= p.vol0 * (tau * grad_w);
      }
    }
  }

  // ── Pass 2: velocity update + gravity + BCs ───────────────────────────────
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      auto &node = grid_[gridIdx(i, j)];
      if (node.mass < 1e-10f)
        continue;

      // Recover velocity from momentum (set during P2G)
      node.vel = node.momentum / node.mass;

      // Add gravity to force  (f_grav = mass * g * e_y)
      node.force.y() += node.mass * params_.gravity;

      // Velocity update:  v_new = v + dt * f / m
      // This is symplectic Euler — v is updated here, positions updated later
      // in advect()
      node.vel_new = node.vel + params_.dt * node.force / node.mass;

      // ── Boundary conditions (sticky walls) ───────────────────────────
      if (i < wall && node.vel_new.x() < 0.f)
        node.vel_new.x() = 0.f;
      if (i >= nx - wall && node.vel_new.x() > 0.f)
        node.vel_new.x() = 0.f;
      if (j < wall && node.vel_new.y() < 0.f)
        node.vel_new.y() = 0.f;
      if (j >= ny - wall && node.vel_new.y() > 0.f)
        node.vel_new.y() = 0.f;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  substep_G2P()  — unchanged from Phase 2
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
        Eigen::Vector2f xip = Eigen::Vector2f(ni * dx, nj * dx) - p.pos;
        v_new += w * vi;
        C_new += w * (vi * xip.transpose());
      }
    }

    p.vel = v_new;
    p.C = D_inv * C_new;
    p.F = (Eigen::Matrix2f::Identity() + dt * p.C) * p.F;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  substep_advect()  — unchanged from Phase 2
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
//  step()
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
//  Polyscope
// ─────────────────────────────────────────────────────────────────────────────
void Simulation::buildRenderArrays(
    std::vector<std::array<double, 3>> &pos3d,
    std::vector<std::array<double, 3>> &colors) const {
  pos3d.resize(particles_.size());
  colors.resize(particles_.size());
  for (size_t i = 0; i < particles_.size(); ++i) {
    const auto &p = particles_[i];
    pos3d[i] = {(double)p.pos.x(), (double)p.pos.y(), 0.0};
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
