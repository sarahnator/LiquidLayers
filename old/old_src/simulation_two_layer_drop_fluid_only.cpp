#include "simulation_two_layer_drop_fluid_only.h"
#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include <algorithm>
#include <cmath>
#include <iostream>

TwoLayerDropSimulation::TwoLayerDropSimulation(SimParams params)
    : params_(std::move(params)) {
  params_.computeDerived();
  const size_t n_nodes = static_cast<size_t>(params_.grid_nx * params_.grid_ny);
  grid_.resize(n_nodes);
  grid_phase_num_.assign(n_nodes, 0.f);
  grid_rho_num_.assign(n_nodes, 0.f);
  grid_weight_sum_.assign(n_nodes, 0.f);
  grid_phase_avg_.assign(n_nodes, 0.f);
  grid_rho_avg_.assign(n_nodes, 0.f);
}

const MaterialParams &
TwoLayerDropSimulation::materialParams(MaterialType m) const {
  switch (m) {
  case MaterialType::Water:
    return water_params_;
  case MaterialType::Soil:
    return soil_params_;
  case MaterialType::Sand:
    return sand_params_;
  case MaterialType::Rock:
    return rock_params_;
  }
  return water_params_;
}

MaterialParams &TwoLayerDropSimulation::materialParamsMutable(MaterialType m) {
  switch (m) {
  case MaterialType::Water:
    return water_params_;
  case MaterialType::Soil:
    return soil_params_;
  case MaterialType::Sand:
    return sand_params_;
  case MaterialType::Rock:
    return rock_params_;
  }
  return water_params_;
}

const MaterialRenderStyle &
TwoLayerDropSimulation::materialStyle(MaterialType m) const {
  switch (m) {
  case MaterialType::Water:
    return water_style_;
  case MaterialType::Soil:
    return soil_style_;
  case MaterialType::Sand:
    return sand_style_;
  case MaterialType::Rock:
    return rock_style_;
  }
  return water_style_;
}

MaterialRenderStyle &
TwoLayerDropSimulation::materialStyleMutable(MaterialType m) {
  switch (m) {
  case MaterialType::Water:
    return water_style_;
  case MaterialType::Soil:
    return soil_style_;
  case MaterialType::Sand:
    return sand_style_;
  case MaterialType::Rock:
    return rock_style_;
  }
  return water_style_;
}

std::array<double, 3>
TwoLayerDropSimulation::renderColor(MaterialType m) const {
  const auto &c = materialStyle(m).color;
  return {static_cast<double>(std::clamp(c[0], 0.f, 1.f)),
          static_cast<double>(std::clamp(c[1], 0.f, 1.f)),
          static_cast<double>(std::clamp(c[2], 0.f, 1.f))};
}

void TwoLayerDropSimulation::enforceFluidOnlyMaterialModels() {
  auto force_fluid = [](MaterialParams &mp) {
    mp.model = ConstitutiveModel::WeaklyCompressibleFluid;
    mp.density0 = std::max(mp.density0, 1.f);
    mp.bulk_modulus = std::max(mp.bulk_modulus, 1e-3f);
    mp.gamma = std::clamp(mp.gamma, 1.f, 12.f);
    mp.viscosity = std::max(mp.viscosity, 0.f);
    mp.computeDerived();
  };

  force_fluid(water_params_);
  force_fluid(soil_params_);
  force_fluid(sand_params_);
  force_fluid(rock_params_);
}

float TwoLayerDropSimulation::phaseSign(MaterialType m) const {
  if (m == scene_params_.top_material)
    return 1.f;
  if (m == scene_params_.bottom_material)
    return -1.f;
  return 0.f;
}

// ─────────────────────────────────────────────────────────────────────────────
//  addTwoLayerBlock()
//
//  Create one dropped rectangular block split into two horizontal bands. This
//  is intentionally simpler than the four-layer drop so density inversion is
//  easier to see.
// ─────────────────────────────────────────────────────────────────────────────
void TwoLayerDropSimulation::addTwoLayerBlock(float x_min, float x_max,
                                              float y_min, float y_max,
                                              MaterialType bottom_mat,
                                              MaterialType top_mat) {
  const float dx = params_.dx;
  const int ppc = params_.ppc;
  const float px = dx / static_cast<float>(ppc);
  const float py = dx / static_cast<float>(ppc);
  const float H = y_max - y_min;

  auto jitter = [&](int i, float s) -> float {
    return s * (static_cast<float>((i * 1013904223u + 1664525u) & 0xFFFFu) /
                    65535.f -
                0.5f);
  };

  const float gap = std::max(scene_params_.layer_gap, 0.f);
  const float layer_h = std::max(0.5f * (H - gap), 0.25f * py);

  struct LayerBand {
    MaterialType mat;
    float y0, y1;
  };

  std::vector<LayerBand> bands = {
      {bottom_mat, y_min, y_min + layer_h},
      {top_mat, y_min + layer_h + gap, y_min + 2.f * layer_h + gap},
  };

  int pidx = 0;
  for (const auto &band : bands) {
    const MaterialParams &mp = materialParams(band.mat);
    for (float y = band.y0 + 0.5f * py; y < band.y1; y += py) {
      for (float x = x_min + 0.5f * px; x < x_max; x += px) {
        Particle p;
        p.pos.x() = std::clamp(x + jitter(2 * pidx, 0.15f * px), 0.001f,
                               params_.domain_w - 0.001f);
        p.pos.y() = std::clamp(y + jitter(2 * pidx + 1, 0.15f * py), 0.001f,
                               params_.domain_h - 0.001f);
        p.material = band.mat;
        p.mass = mp.density0 * px * py;
        p.vol0 = px * py;
        p.F = Eigen::Matrix2f::Identity();
        particles_.push_back(p);
        ++pidx;
      }
    }
  }
}

void TwoLayerDropSimulation::initialize() {
  particles_.clear();
  frame_ = 0;

  params_.computeDerived();
  enforceFluidOnlyMaterialModels();

  const float hw = 0.5f * scene_params_.block_w;
  const float hh = 0.5f * scene_params_.block_h;
  addTwoLayerBlock(
      scene_params_.block_center_x - hw, scene_params_.block_center_x + hw,
      scene_params_.block_center_y - hh, scene_params_.block_center_y + hh,
      scene_params_.bottom_material, scene_params_.top_material);

  std::cout << "[Two-layer drop MPM debug] initialized " << particles_.size()
            << " particles";
}

// ─────────────────────────────────────────────────────────────────────────────
//  stressFluid()
//
//  Weakly compressible Newtonian fluid model:
//    - pressure comes from a Tait equation of state
//    - viscous stress is proportional to the symmetric velocity gradient
//
// constitutive equation for isotropic fluid
// (https://en.wikipedia.org/wiki/Newtonian_fluid#Fluid_model):
// Stress: σ = -pI + 2μD
// (Note: In some codes 2 * viscosity is pre-baked into the 'dynamic_viscosity'
// variable)
//
//  This is the same fluid family as the other debug scenes. Any material can be
//  turned into this fluid model from the UI by changing mp.model.
// ─────────────────────────────────────────────────────────────────────────────
Eigen::Matrix2f
TwoLayerDropSimulation::stressFluid(const Particle &p_const,
                                    const MaterialParams &mp) const {
  Particle &p = const_cast<Particle &>(p_const);

  float J = p.F.determinant();
  // J = std::clamp(J, 0.6f, 1.4f);

  // if (!toggles.model_water_tait)
  //   return Eigen::Matrix2f::Zero();

  const float pressure =
      mp.bulk_modulus * (std::exp(-mp.gamma * std::log(J)) - 1.f);
  const float clamped_pressure =
      std::max(-0.1f, pressure); // optional pressure floor to prevent
                                 // blow-up on large compression
  Eigen::Matrix2f tau = -J * clamped_pressure * Eigen::Matrix2f::Identity();

  if (toggles.enable_viscosity && mp.viscosity > 0.f)
    tau += J * mp.viscosity * (p.C + p.C.transpose());

  return tau;
}

Eigen::Matrix2f
TwoLayerDropSimulation::stressFixedCorotated(const Particle &p,
                                             const MaterialParams &mp) const {
  Eigen::Matrix2f R, S;
  polarDecompose2x2(p.F, R, S);
  const float J = std::clamp(p.F.determinant(), 0.2f, 5.f);
  return 2.f * mp.mu * (p.F - R) * p.F.transpose() +
         mp.lambda_lame * (J - 1.f) * J * Eigen::Matrix2f::Identity();
}

Eigen::Matrix2f
TwoLayerDropSimulation::stressDruckerPrager(const Particle &p,
                                            const MaterialParams &mp) const {
  return stressFixedCorotated(p, mp);
}

Eigen::Matrix2f
TwoLayerDropSimulation::kirchhoffStress(const Particle &p) const {
  const MaterialParams &mp = materialParams(p.material);

  switch (mp.model) {
  case ConstitutiveModel::WeaklyCompressibleFluid:
    return stressFluid(p, mp);

  case ConstitutiveModel::FixedCorotated:
    return stressFixedCorotated(p, mp);

  case ConstitutiveModel::DruckerPrager:
    return stressDruckerPrager(p, mp);
  }
  return Eigen::Matrix2f::Zero();
}

void TwoLayerDropSimulation::projectDruckerPrager(
    Particle &p, const MaterialParams &mp) const {
  Eigen::Matrix2f U, V;
  Eigen::Vector2f sigma_vec;
  svd2x2(p.F, U, sigma_vec, V);
  sigma_vec(0) = std::max(sigma_vec(0), 0.05f);
  sigma_vec(1) = std::max(sigma_vec(1), 0.05f);

  Eigen::Vector2f eps(std::log(sigma_vec(0)), std::log(sigma_vec(1)));
  const float tr_eps = eps(0) + eps(1);
  const Eigen::Vector2f eps_dev =
      eps - (tr_eps / 2.f) * Eigen::Vector2f::Ones();
  const float dev_norm = eps_dev.norm();

  const bool in_tension = (tr_eps >= 0.f);
  const float yield_value = dev_norm + mp.alpha_dp * tr_eps;
  const bool elastic = (!in_tension && yield_value <= 0.f);
  if (elastic)
    return;

  Eigen::Vector2f eps_new;
  if (in_tension || dev_norm < 1e-10f) {
    eps_new = Eigen::Vector2f::Zero();
  } else {
    const float scale = -mp.alpha_dp * tr_eps / dev_norm;
    eps_new = scale * eps_dev + (tr_eps / 2.f) * Eigen::Vector2f::Ones();
  }
  const Eigen::Vector2f sigma_new(std::exp(eps_new(0)), std::exp(eps_new(1)));
  p.F = U * sigma_new.asDiagonal() * V.transpose();
}

void TwoLayerDropSimulation::clearGrid() {
  for (size_t idx = 0; idx < grid_.size(); ++idx) {
    auto &n = grid_[idx];
    n.mass = 0.f;
    n.momentum = n.vel = n.vel_new = n.force = Eigen::Vector2f::Zero();
    grid_phase_num_[idx] = 0.f;
    grid_rho_num_[idx] = 0.f;
    grid_weight_sum_[idx] = 0.f;
    grid_phase_avg_[idx] = 0.f;
    grid_rho_avg_[idx] = 0.f;
  }
}

void TwoLayerDropSimulation::substep_P2G() {
  const float dx = params_.dx;
  const int nx = params_.grid_nx, ny = params_.grid_ny;
  for (const auto &p : particles_) {
    WeightStencil ws(p.pos, dx, nx, ny);
    const MaterialParams &mp = materialParams(p.material);
    const float s = phaseSign(p.material);
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) {
        const int ni = ws.base_i + a, nj = ws.base_j + b;
        if (!inGrid(ni, nj))
          continue;
        const float w = ws.weight(a, b);
        const Eigen::Vector2f xip = Eigen::Vector2f(ni * dx, nj * dx) - p.pos;
        const int idx = gridIdx(ni, nj);
        auto &node = grid_[idx];
        node.mass += w * p.mass;
        node.momentum += w * p.mass * (p.vel + p.C * xip);

        // Build smoothed scalar fields used by the optional helper forces.
        // These are visualization-oriented helpers for the two-layer test, not
        // part of the core MPM update.
        grid_weight_sum_[idx] += w;
        grid_phase_num_[idx] += w * s;
        grid_rho_num_[idx] += w * mp.density0;
      }
  }
  finalizeHelperGridScalars();
}

void TwoLayerDropSimulation::finalizeHelperGridScalars() {
  for (size_t idx = 0; idx < grid_.size(); ++idx) {
    const float wsum = grid_weight_sum_[idx];
    if (wsum > 1e-10f) {
      grid_phase_avg_[idx] = grid_phase_num_[idx] / wsum;
      grid_rho_avg_[idx] = grid_rho_num_[idx] / wsum;
    } else {
      grid_phase_avg_[idx] = 0.f;
      grid_rho_avg_[idx] = 0.f;
    }
  }
}

float TwoLayerDropSimulation::sampleLocalRestDensity(const Particle &p) const {
  const float dx = params_.dx;
  const int nx = params_.grid_nx, ny = params_.grid_ny;
  WeightStencil ws(p.pos, dx, nx, ny);
  float rho_loc = 0.f;
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) {
      const int ni = ws.base_i + a, nj = ws.base_j + b;
      if (!inGrid(ni, nj))
        continue;
      rho_loc += ws.weight(a, b) * grid_rho_avg_[gridIdx(ni, nj)];
    }
  return rho_loc;
}

Eigen::Vector2f
TwoLayerDropSimulation::samplePhaseGradient(const Particle &p) const {
  const float dx = params_.dx;
  const int nx = params_.grid_nx, ny = params_.grid_ny;
  WeightStencil ws(p.pos, dx, nx, ny);
  Eigen::Vector2f grad_phi = Eigen::Vector2f::Zero();
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) {
      const int ni = ws.base_i + a, nj = ws.base_j + b;
      if (!inGrid(ni, nj))
        continue;
      grad_phi += grid_phase_avg_[gridIdx(ni, nj)] * ws.weightGrad(a, b, dx);
    }
  return grad_phi;
}

void TwoLayerDropSimulation::scatterHelperBodyForce(
    const Particle &p, const Eigen::Vector2f &accel) {
  const float dx = params_.dx;
  const int nx = params_.grid_nx, ny = params_.grid_ny;
  WeightStencil ws(p.pos, dx, nx, ny);
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) {
      const int ni = ws.base_i + a, nj = ws.base_j + b;
      if (!inGrid(ni, nj))
        continue;
      grid_[gridIdx(ni, nj)].force += ws.weight(a, b) * p.mass * accel;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  substep_volumeRecompute()
//
//  Phase 5 addition.  Called immediately after P2G (when grid has fresh mass).
//
//  Problem it solves:
//    vol0 is set at initialization to px*py and never updated.  As particles
//    spread out near the free surface, the force formula  f = -vol0*tau*grad_w
//    underestimates the volume each particle actually represents, so
//    free-surface particles generate too little pressure to push their
//    neighbours away and cluster into visible "blobs" before vanishing.
//
//  Fix:
//    After P2G, the grid node at (i,j) holds:
//        node.mass  = sum_p  w_ip * m_p         (units: kg)
//    The density the grid "sees" at node (i,j) is:
//        rho_i      = node.mass / dx^2           (units: kg/m^2 in 2D)
//    We gather this density back to each particle using the same weights:
//        rho_p      = sum_i  w_ip * (node.mass / dx^2)
//    Then the particle's current volume is:
//        vol_p      = m_p / rho_p
//    This vol_p replaces vol0 for this step's stress force scatter.
//
//  Why do we use node.mass/dx^2 and not node.mass directly?
//    node.mass has units of kg.  To get density (kg/m^2) we divide by the
//    cell area dx^2.  This is the same conversion you'd do in SPH when
//    estimating density from neighbour masses.
//
//  Stability note:  we clamp rho_p away from zero to avoid division by zero
//    at the free surface where some grid nodes genuinely have zero mass.
//    The minimum density is set to 10% of the particle's own rest density.
// ─────────────────────────────────────────────────────────────────────────────
void TwoLayerDropSimulation::substep_volumeRecompute() {
  const float dx = params_.dx, dx2 = dx * dx;
  const int nx = params_.grid_nx, ny = params_.grid_ny;

  for (auto &p : particles_) {
    const MaterialParams &mp = materialParams(p.material);

    // Gather density estimate from grid
    float rho_p = 0.f;
    WeightStencil ws(p.pos, dx, nx, ny);
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) {
        int ni = ws.base_i + a, nj = ws.base_j + b;
        if (!inGrid(ni, nj))
          continue;
        float w = ws.weight(a, b);
        // node.mass / dx^2 is the grid-estimated mass density at this node
        rho_p += w * (grid_[gridIdx(ni, nj)].mass / dx2);
      }

    // Clamp to at least 10% of rest density to avoid singularities
    float rho_min = 0.1f * mp.density0;
    rho_p = std::max(rho_p, rho_min);

    // Update volume estimate
    p.vol0 = p.mass / rho_p;
  }
}

void TwoLayerDropSimulation::substep_gridUpdate() {
  const float dx = params_.dx, dt = params_.dt;
  const int nx = params_.grid_nx, ny = params_.grid_ny, wall = 2;

  if (toggles.enable_stress) {
    for (const auto &p : particles_) {
      const Eigen::Matrix2f tau = kirchhoffStress(p);
      WeightStencil ws(p.pos, dx, nx, ny);
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
          const int ni = ws.base_i + a, nj = ws.base_j + b;
          if (!inGrid(ni, nj))
            continue;
          grid_[gridIdx(ni, nj)].force -=
              p.vol0 * (tau * ws.weightGrad(a, b, dx));
        }
    }
  }

  // Optional helper forces for the density-inversion demo.
  //
  // 1) Buoyancy-like helper:
  //      a_b = k_b * |g| * ((rho_p - rho_loc) / rho_loc) * g_hat
  //    where rho_loc is the smoothed neighborhood-average rest density.
  //    Heavy particles in a light neighborhood are nudged downward; light
  //    particles in a heavy neighborhood are nudged upward.
  //
  // 2) Phase-separation helper:
  //      a_s = k_s * |g| * dx * s_p * grad(phi)
  //    where phi is a smoothed phase field with values -1 (bottom fluid) and
  //    +1 (top fluid), and s_p is the particle's phase sign. This pushes the
  //    top phase up the phase gradient and the bottom phase down it.
  if (toggles.enable_buoyancy_helper ||
      toggles.enable_phase_separation_helper) {
    const float gmag = std::abs(params_.gravity);
    const Eigen::Vector2f ghat(0.f, params_.gravity < 0.f ? -1.f : 1.f);

    for (const auto &p : particles_) {
      const MaterialParams &mp = materialParams(p.material);
      const float rho_p = mp.density0;
      const float rho_loc = std::max(sampleLocalRestDensity(p), 1.f);
      const float s = phaseSign(p.material);

      Eigen::Vector2f accel = Eigen::Vector2f::Zero();

      if (toggles.enable_buoyancy_helper) {
        const float contrast =
            std::clamp((rho_p - rho_loc) / rho_loc, -1.f, 1.f);
        accel += scene_params_.buoyancy_strength * gmag * contrast * ghat;
      }

      if (toggles.enable_phase_separation_helper && std::abs(s) > 0.f) {
        const Eigen::Vector2f grad_phi = samplePhaseGradient(p);
        accel +=
            scene_params_.phase_separation_strength * gmag * dx * s * grad_phi;
      }

      if (accel.squaredNorm() > 0.f)
        scatterHelperBodyForce(p, accel);
    }
  }

  for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
      auto &node = grid_[gridIdx(i, j)];
      if (node.mass < 1e-10f)
        continue;

      node.vel = node.momentum / node.mass;
      if (toggles.enable_gravity)
        node.force.y() += node.mass * params_.gravity;

      node.vel_new = node.vel + dt * node.force / node.mass;

      // Enforce slip boundary conditions by zeroing normal velocity at the
      // walls
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

void TwoLayerDropSimulation::substep_G2P() {
  const float dx = params_.dx, D_inv = params_.D_inv, dt = params_.dt;
  const int nx = params_.grid_nx, ny = params_.grid_ny;

  for (auto &p : particles_) {
    WeightStencil ws(p.pos, dx, nx, ny);
    Eigen::Vector2f v_new = Eigen::Vector2f::Zero();
    Eigen::Matrix2f C_new = Eigen::Matrix2f::Zero();
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) {
        const int ni = ws.base_i + a, nj = ws.base_j + b;
        if (!inGrid(ni, nj))
          continue;
        const float w = ws.weight(a, b);
        const Eigen::Vector2f &vi = grid_[gridIdx(ni, nj)].vel_new;
        const Eigen::Vector2f xip = Eigen::Vector2f(ni * dx, nj * dx) - p.pos;
        v_new += w * vi;
        C_new += w * (vi * xip.transpose());
      }
    p.vel = v_new;
    p.C = D_inv * C_new;
    p.F = (Eigen::Matrix2f::Identity() + dt * p.C) * p.F;

    const MaterialParams &mp = materialParams(p.material);
    if (mp.model == ConstitutiveModel::DruckerPrager) {
      projectDruckerPrager(p, mp);
    }
  }
}

void TwoLayerDropSimulation::substep_advect() {
  const float dt = params_.dt, W = params_.domain_w, H = params_.domain_h;
  for (auto &p : particles_) {
    p.pos += dt * p.vel;
    if (toggles.bc_left)
      p.pos.x() = std::max(p.pos.x(), 0.001f);
    if (toggles.bc_right)
      p.pos.x() = std::min(p.pos.x(), W - 0.001f);
    if (toggles.bc_bottom)
      p.pos.y() = std::max(p.pos.y(), 0.001f);
    if (toggles.bc_top)
      p.pos.y() = std::min(p.pos.y(), H - 0.001f);
  }
}

void TwoLayerDropSimulation::step() {
  clearGrid();
  substep_P2G();
  if (toggles.enable_vol_recompute)
    substep_volumeRecompute();
  substep_gridUpdate();
  substep_G2P();
  substep_advect();
  ++frame_;
}

void TwoLayerDropSimulation::buildRenderArrays(
    std::vector<std::array<double, 3>> &pos3d,
    std::vector<std::array<double, 3>> &colors) const {
  pos3d.resize(particles_.size());
  colors.resize(particles_.size());
  for (size_t i = 0; i < particles_.size(); ++i) {
    const auto &p = particles_[i];
    pos3d[i] = {(double)p.pos.x(), (double)p.pos.y(), 0.0};
    colors[i] = renderColor(p.material);
  }
}

void TwoLayerDropSimulation::registerPolyscope() {
  std::vector<std::array<double, 3>> pos3d, colors;
  buildRenderArrays(pos3d, colors);
  auto *cloud = polyscope::registerPointCloud(kCloudName, pos3d);
  cloud->setPointRadius(0.0035);
  cloud->setPointRenderMode(polyscope::PointRenderMode::Sphere); // or Quad
  cloud->setMaterial("flat"); // exact flat RGB
  cloud->addColorQuantity("material_color", colors)->setEnabled(true);
}

void TwoLayerDropSimulation::updatePolyscope() {
  std::vector<std::array<double, 3>> pos3d, colors;
  buildRenderArrays(pos3d, colors);
  auto *cloud = polyscope::getPointCloud(kCloudName);
  cloud->updatePointPositions(pos3d);
  cloud->addColorQuantity("material_color", colors)->setEnabled(true);
}
