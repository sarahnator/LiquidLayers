#include "simulation.h"
#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include <algorithm>
#include <cmath>
#include <iostream>

Simulation::Simulation(SimParams params) : params_(std::move(params)) {
  params_.computeDerived();
  grid_.resize(params_.grid_nx * params_.grid_ny);
}

const MaterialParams &Simulation::materialParams(MaterialType m) const {
  switch (m) {
  case MaterialType::Water:
    return params_.water_params;
  case MaterialType::Soil:
    return params_.soil_params;
  case MaterialType::Sand:
    return params_.sand_params;
  case MaterialType::Rock:
    return params_.rock_params;
  }
  return params_.water_params;
}

MaterialParams &Simulation::materialParamsMutable(MaterialType m) {
  switch (m) {
  case MaterialType::Water:
    return params_.water_params;
  case MaterialType::Soil:
    return params_.soil_params;
  case MaterialType::Sand:
    return params_.sand_params;
  case MaterialType::Rock:
    return params_.rock_params;
  }
  return params_.water_params;
}

// ─────────────────────────────────────────────────────────────────────────────
//  initialize()
// ─────────────────────────────────────────────────────────────────────────────
void Simulation::initialize() {
  particles_.clear();
  frame_ = 0;
  const float W = params_.domain_w, H = params_.domain_h,
              pct = params_.layer_pct, dx = params_.dx;
  const int ppc = params_.ppc;

  struct Layer {
    MaterialType mat;
    float y_lo, y_hi;
  };
  // Default: density-correct order (rock bottom, water top)
  std::vector<Layer> layers = {
      {MaterialType::Rock, 0.f * H * pct, 1.f * H * pct},
      {MaterialType::Sand, 1.f * H * pct, 2.f * H * pct},
      {MaterialType::Soil, 2.f * H * pct, 3.f * H * pct},
      {MaterialType::Water, 3.f * H * pct, 4.f * H * pct},
  };

  // // Reverse density order (physically unstable, should stratify)
  // std::vector<Layer> layers = {
  //     {MaterialType::Water, 0.f * H * pct, 1.f * H * pct},
  //     {MaterialType::Soil, 1.f * H * pct, 2.f * H * pct},
  //     {MaterialType::Sand, 2.f * H * pct, 3.f * H * pct},
  //     {MaterialType::Rock, 3.f * H * pct, 4.f * H * pct},
  // };

  const float px = dx / ppc, py = dx / ppc;
  auto jitter = [&](int i, float s) -> float {
    return s *
           (static_cast<float>((i * 1013904223 + 1664525) & 0xFFFF) / 65535.f -
            0.5f);
  };
  int pidx = 0;
  for (const auto &l : layers) {
    const MaterialParams &mp = materialParams(l.mat);
    for (float y = l.y_lo + py * 0.5f; y < l.y_hi; y += py)
      for (float x = px * 0.5f; x < W; x += px) {
        Particle p;
        p.pos.x() =
            std::clamp(x + jitter(pidx * 2, px * 0.3f), 0.001f, W - 0.001f);
        p.pos.y() =
            std::clamp(y + jitter(pidx * 2 + 1, py * 0.3f), 0.001f, H - 0.001f);
        p.material = l.mat;
        p.mass = mp.density0 * px * py;
        p.vol0 = px * py;
        particles_.push_back(p);
        ++pidx;
      }
  }
  std::cout << "[MPM debug] " << particles_.size() << " particles\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  stressFluid()
//
//  Toggle checks:
//    toggles.model_water_tait   — if off, returns zero (pressureless gas)
//    toggles.enable_viscosity   — if off, skips viscous term
// ─────────────────────────────────────────────────────────────────────────────
Eigen::Matrix2f Simulation::stressFluid(const Particle &p_const,
                                        const MaterialParams &mp) const {
  Particle &p = const_cast<Particle &>(p_const);

  float J = p.F.determinant();
  J = std::clamp(J, 0.6f, 1.4f);

  if (!toggles.model_water_tait)
    return Eigen::Matrix2f::Zero(); // Phase 1/2 fallback: no pressure

  float pressure = mp.bulk_modulus * (std::exp(-mp.gamma * std::log(J)) - 1.f);
  Eigen::Matrix2f tau = -J * pressure * Eigen::Matrix2f::Identity();

  if (toggles.enable_viscosity && mp.viscosity > 0.f)
    tau += J * mp.viscosity * (p.C + p.C.transpose());

  return tau;
}

// ─────────────────────────────────────────────────────────────────────────────
//  stressFixedCorotated()
//
//  Toggle: toggles.model_soil_elastic / model_rock_elastic (checked in
//  dispatcher)
// ─────────────────────────────────────────────────────────────────────────────
Eigen::Matrix2f
Simulation::stressFixedCorotated(const Particle &p,
                                 const MaterialParams &mp) const {
  Eigen::Matrix2f R, S;
  polarDecompose2x2(p.F, R, S);
  float J = std::clamp(p.F.determinant(), 0.2f, 5.f);
  return 2.f * mp.mu * (p.F - R) * p.F.transpose() +
         mp.lambda_lame * (J - 1.f) * J * Eigen::Matrix2f::Identity();
}

// ─────────────────────────────────────────────────────────────────────────────
//  stressDruckerPrager()  — elastic stress only; projection handled in G2P
// ─────────────────────────────────────────────────────────────────────────────
Eigen::Matrix2f
Simulation::stressDruckerPrager(const Particle &p,
                                const MaterialParams &mp) const {
  return stressFixedCorotated(p, mp);
}

// ─────────────────────────────────────────────────────────────────────────────
//  kirchhoffStress()  — dispatcher with per-material toggle checks
// ─────────────────────────────────────────────────────────────────────────────
Eigen::Matrix2f Simulation::kirchhoffStress(const Particle &p) const {
  const MaterialParams &mp = materialParams(p.material);

  switch (mp.model) {
  case ConstitutiveModel::WeaklyCompressibleFluid:
    // stressFluid handles its own toggle internally
    return stressFluid(p, mp);

  case ConstitutiveModel::FixedCorotated:
    if (p.material == MaterialType::Soil && !toggles.model_soil_elastic)
      return Eigen::Matrix2f::Zero();
    if (p.material == MaterialType::Rock && !toggles.model_rock_elastic)
      return Eigen::Matrix2f::Zero();
    return stressFixedCorotated(p, mp);

  case ConstitutiveModel::DruckerPrager:
    // If plasticity is disabled, we still compute elastic stress —
    // sand without yield surface behaves as a soft elastic solid,
    // which is useful to compare against the full plastic version.
    return stressDruckerPrager(p, mp);
  }
  return Eigen::Matrix2f::Zero();
}

// ─────────────────────────────────────────────────────────────────────────────
//  projectDruckerPrager()
// ─────────────────────────────────────────────────────────────────────────────
void Simulation::projectDruckerPrager(Particle &p,
                                      const MaterialParams &mp) const {
  Eigen::Matrix2f U, V;
  Eigen::Vector2f sigma_vec;
  svd2x2(p.F, U, sigma_vec, V);
  sigma_vec(0) = std::max(sigma_vec(0), 0.05f);
  sigma_vec(1) = std::max(sigma_vec(1), 0.05f);

  Eigen::Vector2f eps(std::log(sigma_vec(0)), std::log(sigma_vec(1)));
  float tr_eps = eps(0) + eps(1);
  Eigen::Vector2f eps_dev = eps - (tr_eps / 2.f) * Eigen::Vector2f::Ones();
  float dev_norm = eps_dev.norm();

  bool in_tension = (tr_eps >= 0.f);
  float yield_value = dev_norm + mp.alpha_dp * tr_eps;
  bool elastic = (!in_tension && yield_value <= 0.f);
  if (elastic)
    return;

  Eigen::Vector2f eps_new;
  if (in_tension || dev_norm < 1e-10f) {
    eps_new = Eigen::Vector2f::Zero();
  } else {
    float scale = -mp.alpha_dp * tr_eps / dev_norm;
    eps_new = scale * eps_dev + (tr_eps / 2.f) * Eigen::Vector2f::Ones();
  }
  Eigen::Vector2f sigma_new(std::exp(eps_new(0)), std::exp(eps_new(1)));
  p.F = U * sigma_new.asDiagonal() * V.transpose();
}

// ─────────────────────────────────────────────────────────────────────────────
//  clearGrid()
// ─────────────────────────────────────────────────────────────────────────────
void Simulation::clearGrid() {
  for (auto &n : grid_) {
    n.mass = 0.f;
    n.momentum = n.vel = n.vel_new = n.force = Eigen::Vector2f::Zero();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  substep_P2G()
// ─────────────────────────────────────────────────────────────────────────────
void Simulation::substep_P2G() {
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
        Eigen::Vector2f xip = Eigen::Vector2f(ni * dx, nj * dx) - p.pos;
        auto &node = grid_[gridIdx(ni, nj)];
        node.mass += w * p.mass;
        node.momentum += w * p.mass * (p.vel + p.C * xip);
      }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  substep_gridUpdate()
//
//  Toggle checks:
//    toggles.enable_stress  — scatter stress forces to the grid
//    toggles.enable_gravity — add gravitational acceleration
//    toggles.bc_*           — enforce each wall boundary condition
// ─────────────────────────────────────────────────────────────────────────────
void Simulation::substep_gridUpdate() {
  const float dx = params_.dx, dt = params_.dt;
  const int nx = params_.grid_nx, ny = params_.grid_ny, wall = 2;

  // Pass 1: stress forces (Phase 3+)
  if (toggles.enable_stress) {
    for (const auto &p : particles_) {
      Eigen::Matrix2f tau = kirchhoffStress(p);
      WeightStencil ws(p.pos, dx, nx, ny);
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
          int ni = ws.base_i + a, nj = ws.base_j + b;
          if (!inGrid(ni, nj))
            continue;
          grid_[gridIdx(ni, nj)].force -=
              p.vol0 * (tau * ws.weightGrad(a, b, dx));
        }
    }
  }

  // Pass 2: velocity update
  for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
      auto &node = grid_[gridIdx(i, j)];
      if (node.mass < 1e-10f)
        continue;

      node.vel = node.momentum / node.mass;

      // Gravity (Phase 3+)
      if (toggles.enable_gravity)
        node.force.y() += node.mass * params_.gravity;

      node.vel_new = node.vel + dt * node.force / node.mass;

      // Boundary conditions — each wall independently toggleable
      if (toggles.bc_left && i < wall && node.vel_new.x() < 0.f)
        node.vel_new.x() = 0.f;
      if (toggles.bc_right && i >= nx - wall && node.vel_new.x() > 0.f)
        node.vel_new.x() = 0.f;
      if (toggles.bc_bottom && j < wall && node.vel_new.y() < 0.f)
        node.vel_new.y() = 0.f;
      if (toggles.bc_top && j >= ny - wall && node.vel_new.y() > 0.f)
        node.vel_new.y() = 0.f;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  substep_G2P()
//
//  Toggle checks:
//    toggles.model_sand_plastic — Drucker-Prager projection for sand
// ─────────────────────────────────────────────────────────────────────────────
void Simulation::substep_G2P() {
  const float dx = params_.dx, D_inv = params_.D_inv, dt = params_.dt;
  const int nx = params_.grid_nx, ny = params_.grid_ny;

  for (auto &p : particles_) {
    WeightStencil ws(p.pos, dx, nx, ny);
    Eigen::Vector2f v_new = Eigen::Vector2f::Zero();
    Eigen::Matrix2f C_new = Eigen::Matrix2f::Zero();
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) {
        int ni = ws.base_i + a, nj = ws.base_j + b;
        if (!inGrid(ni, nj))
          continue;
        float w = ws.weight(a, b);
        const Eigen::Vector2f &vi = grid_[gridIdx(ni, nj)].vel_new;
        Eigen::Vector2f xip = Eigen::Vector2f(ni * dx, nj * dx) - p.pos;
        v_new += w * vi;
        C_new += w * (vi * xip.transpose());
      }
    p.vel = v_new;
    p.C = D_inv * C_new;
    p.F = (Eigen::Matrix2f::Identity() + dt * p.C) * p.F;

    const MaterialParams &mp = materialParams(p.material);

    if (mp.model == ConstitutiveModel::DruckerPrager) {
      if (toggles.model_sand_plastic)
        projectDruckerPrager(p, mp);
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  substep_advect()
// ─────────────────────────────────────────────────────────────────────────────
void Simulation::substep_advect() {
  const float dt = params_.dt, W = params_.domain_w, H = params_.domain_h;
  for (auto &p : particles_) {
    p.pos += dt * p.vel;
    p.pos.x() = std::clamp(p.pos.x(), 0.001f, W - 0.001f);
    p.pos.y() = std::clamp(p.pos.y(), 0.001f, H - 0.001f);
  }
}

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
    pos3d[i] = {(double)p.pos.x(), (double)p.pos.y(), 0.};
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