#include "simulation_many_blob_demo.h"
#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include <algorithm>
#include <cmath>
#include <iostream>

ManyBlobSimulation::ManyBlobSimulation(SimParams params) : params_(std::move(params)) {
  params_.computeDerived();
  const size_t n_nodes = static_cast<size_t>(params_.grid_nx * params_.grid_ny);
  grid_.resize(n_nodes);
  grid_phase_num_.resize(n_nodes, 0.f);
  grid_rho_num_.resize(n_nodes, 0.f);
  grid_weight_sum_.resize(n_nodes, 0.f);
  grid_phase_avg_.resize(n_nodes, 0.f);
  grid_rho_avg_.resize(n_nodes, 0.f);
  grid_phase_lap_.resize(n_nodes, 0.f);
}

const MaterialParams &ManyBlobSimulation::materialParams(MaterialType m) const {
  return (m == MaterialType::Water) ? water_params_ : rock_params_;
}

MaterialParams &ManyBlobSimulation::materialParamsMutable(MaterialType m) {
  return (m == MaterialType::Water) ? water_params_ : rock_params_;
}

void ManyBlobSimulation::addBlob(const Eigen::Vector2f &center, float radius,
                                 MaterialType mat, int blob_index) {
  const float dx = params_.dx;
  const int ppc = params_.ppc;
  const float px = dx / static_cast<float>(ppc);
  const float py = dx / static_cast<float>(ppc);
  const MaterialParams &mp = materialParams(mat);

  auto jitter = [&](int i, float s) -> float {
    return s *
           (static_cast<float>((i * 1013904223u + 1664525u) & 0xFFFFu) /
                65535.f -
            0.5f);
  };

  // Give each blob a small tangential swirl plus a downward component. This
  // makes the scene livelier and helps break symmetry in a way that is closer
  // to an appearance-first liquid toy.
  const Eigen::Vector2f omega_center = center;

  int pidx = 0;
  for (float y = center.y() - radius; y <= center.y() + radius; y += py) {
    for (float x = center.x() - radius; x <= center.x() + radius; x += px) {
      Eigen::Vector2f pos(x, y);
      if ((pos - center).squaredNorm() > radius * radius)
        continue;

      Particle p;
      p.pos.x() = std::clamp(x + jitter(2 * pidx + 17 * blob_index, 0.18f * px),
                             0.001f, params_.domain_w - 0.001f);
      p.pos.y() = std::clamp(y + jitter(2 * pidx + 1 + 17 * blob_index, 0.18f * py),
                             0.001f, params_.domain_h - 0.001f);
      p.material = mat;
      p.mass = mp.density0 * px * py;
      p.vol0 = px * py;
      p.F = Eigen::Matrix2f::Identity();

      Eigen::Vector2f rel = p.pos - omega_center;
      Eigen::Vector2f tangent(-rel.y(), rel.x());
      if (tangent.norm() > 1e-8f)
        tangent.normalize();
      p.vel = scene_params_.initial_swirl_speed * tangent +
              Eigen::Vector2f(0.f, scene_params_.initial_downward_speed);

      particles_.push_back(p);
      ++pidx;
    }
  }
}

void ManyBlobSimulation::initialize() {
  particles_.clear();
  frame_ = 0;
  params_.computeDerived();
  water_params_.computeDerived();
  rock_params_.computeDerived();

  int blob_index = 0;
  for (int j = 0; j < scene_params_.num_blobs_y; ++j) {
    for (int i = 0; i < scene_params_.num_blobs_x; ++i) {
      Eigen::Vector2f c(scene_params_.blob_center_x + i * scene_params_.blob_spacing_x,
                        scene_params_.blob_center_y - j * scene_params_.blob_spacing_y);
      const bool use_water = ((i + j) % 2 == 0);
      addBlob(c, scene_params_.blob_radius,
              use_water ? MaterialType::Water : MaterialType::Rock,
              blob_index);
      ++blob_index;
    }
  }

  std::cout << "[Many blob MPM demo] initialized " << particles_.size()
            << " particles\n";
}

Eigen::Matrix2f ManyBlobSimulation::stressFluid(const Particle &p_const,
                                                const MaterialParams &mp) const {
  Particle &p = const_cast<Particle &>(p_const);
  float J = p.F.determinant();
  J = std::clamp(J, 0.55f, 1.55f);

  if (toggles.model_water_freset)
    p.F = std::sqrt(J) * Eigen::Matrix2f::Identity();

  if (!toggles.model_water_tait)
    return Eigen::Matrix2f::Zero();

  const float pressure = mp.bulk_modulus * (std::exp(-mp.gamma * std::log(J)) - 1.f);
  Eigen::Matrix2f tau = -J * pressure * Eigen::Matrix2f::Identity();

  if (toggles.enable_viscosity && mp.viscosity > 0.f)
    tau += J * mp.viscosity * (p.C + p.C.transpose());

  return tau;
}

Eigen::Matrix2f ManyBlobSimulation::kirchhoffStress(const Particle &p) const {
  return stressFluid(p, materialParams(p.material));
}

void ManyBlobSimulation::clearGrid() {
  for (auto &n : grid_) {
    n.mass = 0.f;
    n.momentum = n.vel = n.vel_new = n.force = Eigen::Vector2f::Zero();
  }
  std::fill(grid_phase_num_.begin(), grid_phase_num_.end(), 0.f);
  std::fill(grid_rho_num_.begin(), grid_rho_num_.end(), 0.f);
  std::fill(grid_weight_sum_.begin(), grid_weight_sum_.end(), 0.f);
  std::fill(grid_phase_avg_.begin(), grid_phase_avg_.end(), 0.f);
  std::fill(grid_rho_avg_.begin(), grid_rho_avg_.end(), 0.f);
  std::fill(grid_phase_lap_.begin(), grid_phase_lap_.end(), 0.f);
}

void ManyBlobSimulation::substep_P2G() {
  const float dx = params_.dx;
  const int nx = params_.grid_nx;
  const int ny = params_.grid_ny;

  for (const auto &p : particles_) {
    WeightStencil ws(p.pos, dx, nx, ny);
    const float phase_sign = (p.material == MaterialType::Water) ? -1.f : 1.f;
    const float rho0 = materialParams(p.material).density0;

    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        const int ni = ws.base_i + a;
        const int nj = ws.base_j + b;
        if (!inGrid(ni, nj))
          continue;

        const float w = ws.weight(a, b);
        const Eigen::Vector2f xip = Eigen::Vector2f(ni * dx, nj * dx) - p.pos;
        const int idx = gridIdx(ni, nj);
        auto &node = grid_[idx];
        node.mass += w * p.mass;
        node.momentum += w * p.mass * (p.vel + p.C * xip);

        grid_phase_num_[idx] += w * phase_sign;
        grid_rho_num_[idx] += w * rho0;
        grid_weight_sum_[idx] += w;
      }
    }
  }
}

void ManyBlobSimulation::finalizeHelperGridScalars() {
  const int nx = params_.grid_nx;
  const int ny = params_.grid_ny;
  const float dx = params_.dx;

  for (size_t idx = 0; idx < grid_.size(); ++idx) {
    if (grid_weight_sum_[idx] > 1e-8f) {
      grid_phase_avg_[idx] = grid_phase_num_[idx] / grid_weight_sum_[idx];
      grid_rho_avg_[idx] = grid_rho_num_[idx] / grid_weight_sum_[idx];
    } else {
      grid_phase_avg_[idx] = 0.f;
      grid_rho_avg_[idx] = 0.f;
    }
  }

  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      const int idx = gridIdx(i, j);
      const float c = grid_phase_avg_[idx];
      const float l = inGrid(i - 1, j) ? grid_phase_avg_[gridIdx(i - 1, j)] : c;
      const float r = inGrid(i + 1, j) ? grid_phase_avg_[gridIdx(i + 1, j)] : c;
      const float b = inGrid(i, j - 1) ? grid_phase_avg_[gridIdx(i, j - 1)] : c;
      const float t = inGrid(i, j + 1) ? grid_phase_avg_[gridIdx(i, j + 1)] : c;
      grid_phase_lap_[idx] = (l + r + b + t - 4.f * c) / (dx * dx);
    }
  }
}

float ManyBlobSimulation::sampleLocalRestDensity(const Particle &p) const {
  const float dx = params_.dx;
  const int nx = params_.grid_nx;
  const int ny = params_.grid_ny;
  WeightStencil ws(p.pos, dx, nx, ny);
  float rho = 0.f;
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) {
      const int ni = ws.base_i + a;
      const int nj = ws.base_j + b;
      if (!inGrid(ni, nj))
        continue;
      rho += ws.weight(a, b) * grid_rho_avg_[gridIdx(ni, nj)];
    }
  return rho;
}

Eigen::Vector2f ManyBlobSimulation::samplePhaseGradient(const Particle &p) const {
  const float dx = params_.dx;
  const int nx = params_.grid_nx;
  const int ny = params_.grid_ny;
  WeightStencil ws(p.pos, dx, nx, ny);
  Eigen::Vector2f grad = Eigen::Vector2f::Zero();
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) {
      const int ni = ws.base_i + a;
      const int nj = ws.base_j + b;
      if (!inGrid(ni, nj))
        continue;
      grad += grid_phase_avg_[gridIdx(ni, nj)] * ws.weightGrad(a, b, dx);
    }
  return grad;
}

float ManyBlobSimulation::samplePhaseLaplacian(const Particle &p) const {
  const float dx = params_.dx;
  const int nx = params_.grid_nx;
  const int ny = params_.grid_ny;
  WeightStencil ws(p.pos, dx, nx, ny);
  float lap = 0.f;
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) {
      const int ni = ws.base_i + a;
      const int nj = ws.base_j + b;
      if (!inGrid(ni, nj))
        continue;
      lap += ws.weight(a, b) * grid_phase_lap_[gridIdx(ni, nj)];
    }
  return lap;
}

void ManyBlobSimulation::scatterHelperBodyForce(const Particle &p,
                                                const Eigen::Vector2f &accel) {
  const float dx = params_.dx;
  const int nx = params_.grid_nx;
  const int ny = params_.grid_ny;
  WeightStencil ws(p.pos, dx, nx, ny);
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) {
      const int ni = ws.base_i + a;
      const int nj = ws.base_j + b;
      if (!inGrid(ni, nj))
        continue;
      grid_[gridIdx(ni, nj)].force += ws.weight(a, b) * p.mass * accel;
    }
}

void ManyBlobSimulation::substep_gridUpdate() {
  const float dx = params_.dx;
  const float dt = params_.dt;
  const int nx = params_.grid_nx;
  const int ny = params_.grid_ny;
  const int wall = 2;

  if (toggles.enable_stress) {
    for (const auto &p : particles_) {
      const Eigen::Matrix2f tau = kirchhoffStress(p);
      WeightStencil ws(p.pos, dx, nx, ny);
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
          const int ni = ws.base_i + a;
          const int nj = ws.base_j + b;
          if (!inGrid(ni, nj))
            continue;
          grid_[gridIdx(ni, nj)].force -= p.vol0 * (tau * ws.weightGrad(a, b, dx));
        }
    }
  }

  finalizeHelperGridScalars();

  const float abs_g = std::abs(params_.gravity);
  for (const auto &p : particles_) {
    Eigen::Vector2f accel = Eigen::Vector2f::Zero();
    const float phase_sign = (p.material == MaterialType::Water) ? -1.f : 1.f;

    if (toggles.enable_buoyancy_helper) {
      const float rho_local = std::max(sampleLocalRestDensity(p), 1.f);
      const float rho_self = materialParams(p.material).density0;
      const float contrast = (rho_self - rho_local) / rho_local;
      accel.y() += -scene_params_.buoyancy_strength * abs_g * contrast;
    }

    const Eigen::Vector2f grad_phi = samplePhaseGradient(p);
    if (toggles.enable_phase_separation_helper)
      accel += scene_params_.phase_separation_strength * abs_g * dx * phase_sign * grad_phi;

    if (toggles.enable_surface_tension_helper) {
      const float lap_phi = samplePhaseLaplacian(p);
      accel += -scene_params_.surface_tension_strength * abs_g * dx * dx * lap_phi * grad_phi;
    }

    scatterHelperBodyForce(p, accel);
  }

  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      auto &node = grid_[gridIdx(i, j)];
      if (node.mass < 1e-10f)
        continue;

      node.vel = node.momentum / node.mass;
      if (toggles.enable_gravity)
        node.force.y() += node.mass * params_.gravity;

      node.vel_new = node.vel + dt * node.force / node.mass;

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
}

void ManyBlobSimulation::substep_G2P() {
  const float dx = params_.dx;
  const float D_inv = params_.D_inv;
  const float dt = params_.dt;
  const int nx = params_.grid_nx;
  const int ny = params_.grid_ny;

  for (auto &p : particles_) {
    WeightStencil ws(p.pos, dx, nx, ny);
    Eigen::Vector2f v_new = Eigen::Vector2f::Zero();
    Eigen::Matrix2f C_new = Eigen::Matrix2f::Zero();

    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) {
        const int ni = ws.base_i + a;
        const int nj = ws.base_j + b;
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

    if (toggles.model_water_freset) {
      const float J = std::clamp(p.F.determinant(), 0.6f, 1.4f);
      p.F = std::sqrt(J) * Eigen::Matrix2f::Identity();
    }
  }
}

void ManyBlobSimulation::substep_advect() {
  const float dt = params_.dt;
  const float W = params_.domain_w;
  const float H = params_.domain_h;
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

void ManyBlobSimulation::step() {
  clearGrid();
  substep_P2G();
  substep_gridUpdate();
  substep_G2P();
  substep_advect();
  ++frame_;
}

void ManyBlobSimulation::buildRenderArrays(
    std::vector<std::array<double, 3>> &pos3d,
    std::vector<std::array<double, 3>> &colors) const {
  pos3d.resize(particles_.size());
  colors.resize(particles_.size());
  for (size_t i = 0; i < particles_.size(); ++i) {
    const auto &p = particles_[i];
    pos3d[i] = {(double)p.pos.x(), (double)p.pos.y(), 0.0};
    const auto c = materialColor(p.material);
    colors[i] = {c[0], c[1], c[2]};
  }
}

void ManyBlobSimulation::registerPolyscope() {
  std::vector<std::array<double, 3>> pos3d, colors;
  buildRenderArrays(pos3d, colors);
  auto *cloud = polyscope::registerPointCloud(kCloudName, pos3d);
  cloud->setPointRadius(0.0038);
  cloud->setPointRenderMode(polyscope::PointRenderMode::Sphere);
  cloud->addColorQuantity("material_color", colors)->setEnabled(true);
}

void ManyBlobSimulation::updatePolyscope() {
  std::vector<std::array<double, 3>> pos3d, colors;
  buildRenderArrays(pos3d, colors);
  auto *cloud = polyscope::getPointCloud(kCloudName);
  cloud->updatePointPositions(pos3d);
  cloud->addColorQuantity("material_color", colors)->setEnabled(true);
}
