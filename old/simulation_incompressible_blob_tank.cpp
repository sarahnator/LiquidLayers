#include "simulation_incompressible_blob_tank.h"

#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace {
float clamp01(float x) { return std::clamp(x, 0.f, 1.f); }
}

IncompressibleBlobTankSimulation::IncompressibleBlobTankSimulation(
    SimParams params)
    : params_(std::move(params)) {
  params_.computeDerived();
  const int n = params_.grid_nx * params_.grid_ny;
  grid_.resize(static_cast<size_t>(n));
  pressure_.assign(static_cast<size_t>(n), 0.f);
  pressure_tmp_.assign(static_cast<size_t>(n), 0.f);
  divergence_.assign(static_cast<size_t>(n), 0.f);
  phase_scalar_.assign(static_cast<size_t>(n), 0.f);
  phase_laplacian_.assign(static_cast<size_t>(n), 0.f);
  mass_a_.assign(static_cast<size_t>(n), 0.f);
  mass_b_.assign(static_cast<size_t>(n), 0.f);

  phase_a_params_.model = ConstitutiveModel::WeaklyCompressibleFluid;
  phase_b_params_.model = ConstitutiveModel::WeaklyCompressibleFluid;
}

const MaterialParams& IncompressibleBlobTankSimulation::materialParams(
    MaterialType m) const {
  return (m == MaterialType::Water) ? phase_a_params_ : phase_b_params_;
}

MaterialParams& IncompressibleBlobTankSimulation::materialParamsMutable(
    MaterialType m) {
  return (m == MaterialType::Water) ? phase_a_params_ : phase_b_params_;
}

void IncompressibleBlobTankSimulation::addBlob(const Eigen::Vector2f& center,
                                               float radius,
                                               MaterialType mat,
                                               Eigen::Vector2f initial_vel) {
  const float dx = params_.dx;
  const int ppc = params_.ppc;
  const float px = dx / static_cast<float>(ppc);
  const float py = dx / static_cast<float>(ppc);
  const float r2 = radius * radius;

  const MaterialParams& mp = materialParams(mat);
  int seed = static_cast<int>(particles_.size()) * 17 + 13;
  auto jitter = [&](int k, float s) {
    const unsigned u = static_cast<unsigned>(seed + 1103515245u * (k + 3));
    return s * ((static_cast<float>(u & 0xFFFFu) / 65535.f) - 0.5f);
  };

  for (float y = center.y() - radius; y <= center.y() + radius; y += py) {
    for (float x = center.x() - radius; x <= center.x() + radius; x += px) {
      Eigen::Vector2f q(x, y);
      if ((q - center).squaredNorm() > r2)
        continue;

      Particle p;
      p.pos.x() = std::clamp(x + jitter(seed++, 0.25f * px),
                             scene_params_.tank_xmin + 0.001f,
                             scene_params_.tank_xmax - 0.001f);
      p.pos.y() = std::clamp(y + jitter(seed++, 0.25f * py),
                             scene_params_.tank_ymin + 0.001f,
                             scene_params_.tank_ymax - 0.001f);
      p.vel = initial_vel;
      p.C.setZero();
      p.F.setIdentity();
      p.material = mat;
      p.vol0 = px * py;
      p.mass = mp.density0 * p.vol0;
      particles_.push_back(p);
    }
  }
}

void IncompressibleBlobTankSimulation::addRandomBlobPattern() {
  particles_.clear();

  phase_a_params_.density0 = std::max(scene_params_.phase_a_density, 1.f);
  phase_b_params_.density0 = std::max(scene_params_.phase_b_density, 1.f);

  const float xmin = scene_params_.tank_xmin;
  const float xmax = scene_params_.tank_xmax;
  const float ymin = scene_params_.tank_ymin;
  const float ymax = scene_params_.spawn_ymax;
  const int rows = std::max(scene_params_.blob_rows, 1);
  const int cols = std::max(scene_params_.blob_cols, 1);
  const float r = std::max(scene_params_.blob_radius, 0.6f * params_.dx);
  const float j = std::max(scene_params_.blob_jitter, 0.f);

  const float sx = (xmax - xmin) / static_cast<float>(cols + 1);
  const float sy = (ymax - ymin) / static_cast<float>(rows + 1);

  int id = 0;
  auto hash01 = [&](int n) {
    const unsigned u = static_cast<unsigned>(1664525u * (n + 101) + 1013904223u);
    return static_cast<float>(u & 0xFFFFu) / 65535.f;
  };

  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      const float keep = hash01(13 * row + 19 * col + 5);
      if (keep < scene_params_.void_probability)
        continue;

      Eigen::Vector2f center(
          xmin + (static_cast<float>(col) + 1.f) * sx +
              (hash01(id + 1) - 0.5f) * j * sx,
          ymin + (static_cast<float>(row) + 1.f) * sy +
              (hash01(id + 7) - 0.5f) * j * sy);

      const bool phase_b = ((row + col + (hash01(id + 31) > 0.65f ? 1 : 0)) % 2) == 1;
      const MaterialType mat = phase_b ? MaterialType::Rock : MaterialType::Water;

      // Small swirl + gentle upward/downward bias to avoid a totally static start.
      const float angle = 2.f * static_cast<float>(M_PI) * hash01(id + 17);
      const float speed = 0.15f + 0.20f * hash01(id + 29);
      Eigen::Vector2f v0(speed * std::cos(angle), speed * std::sin(angle));
      v0.y() += 0.1f * (phase_b ? -1.f : 1.f);

      addBlob(center, r * (0.85f + 0.3f * hash01(id + 47)), mat, v0);
      ++id;
    }
  }
}

void IncompressibleBlobTankSimulation::initialize() {
  params_.computeDerived();
  addRandomBlobPattern();
  frame_ = 0;
  time_ = 0.f;
  std::cout << "[Incompressible blob tank] initialized " << particles_.size()
            << " particles\n";
}

void IncompressibleBlobTankSimulation::clearGrid() {
  for (auto& node : grid_) {
    node.mass = 0.f;
    node.momentum.setZero();
    node.vel.setZero();
    node.vel_new.setZero();
    node.force.setZero();
  }
  std::fill(pressure_.begin(), pressure_.end(), 0.f);
  std::fill(pressure_tmp_.begin(), pressure_tmp_.end(), 0.f);
  std::fill(divergence_.begin(), divergence_.end(), 0.f);
  std::fill(phase_scalar_.begin(), phase_scalar_.end(), 0.f);
  std::fill(phase_laplacian_.begin(), phase_laplacian_.end(), 0.f);
  std::fill(mass_a_.begin(), mass_a_.end(), 0.f);
  std::fill(mass_b_.begin(), mass_b_.end(), 0.f);
}

void IncompressibleBlobTankSimulation::substep_P2G() {
  const float dx = params_.dx;
  const int nx = params_.grid_nx;
  const int ny = params_.grid_ny;

  for (const auto& p : particles_) {
    WeightStencil ws(p.pos, dx, nx, ny);
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        const int ni = ws.base_i + a;
        const int nj = ws.base_j + b;
        if (!inGrid(ni, nj))
          continue;

        const float w = ws.weight(a, b);
        const Eigen::Vector2f xip(Eigen::Vector2f(ni * dx, nj * dx) - p.pos);
        auto& node = grid_[gridIdx(ni, nj)];
        node.mass += w * p.mass;
        node.momentum += w * p.mass * (p.vel + p.C * xip);
        if (p.material == MaterialType::Water)
          mass_a_[gridIdx(ni, nj)] += w * p.mass;
        else
          mass_b_[gridIdx(ni, nj)] += w * p.mass;
      }
    }
  }
}

void IncompressibleBlobTankSimulation::finalizePhaseFields() {
  const int nx = params_.grid_nx;
  const int ny = params_.grid_ny;
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      const int idx = gridIdx(i, j);
      const float ma = mass_a_[idx];
      const float mb = mass_b_[idx];
      const float mt = ma + mb;
      if (mt > 1e-8f)
        phase_scalar_[idx] = (mb - ma) / mt;
      else
        phase_scalar_[idx] = 0.f;
    }
  }

  const float inv_dx2 = 1.f / (params_.dx * params_.dx);
  for (int j = 1; j < ny - 1; ++j) {
    for (int i = 1; i < nx - 1; ++i) {
      const int idx = gridIdx(i, j);
      phase_laplacian_[idx] =
          (phase_scalar_[gridIdx(i - 1, j)] + phase_scalar_[gridIdx(i + 1, j)] +
           phase_scalar_[gridIdx(i, j - 1)] + phase_scalar_[gridIdx(i, j + 1)] -
           4.f * phase_scalar_[idx]) *
          inv_dx2;
    }
  }
}

Eigen::Vector2f IncompressibleBlobTankSimulation::samplePhaseGradient(
    const Particle& p) const {
  const float dx = params_.dx;
  const int nx = params_.grid_nx;
  const int ny = params_.grid_ny;
  WeightStencil ws(p.pos, dx, nx, ny);
  Eigen::Vector2f grad = Eigen::Vector2f::Zero();
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      const int ni = ws.base_i + a;
      const int nj = ws.base_j + b;
      if (!inGrid(ni, nj))
        continue;
      grad += phase_scalar_[gridIdx(ni, nj)] * ws.weightGrad(a, b, dx);
    }
  }
  return grad;
}

float IncompressibleBlobTankSimulation::samplePhaseLaplacian(
    const Particle& p) const {
  const float dx = params_.dx;
  const int nx = params_.grid_nx;
  const int ny = params_.grid_ny;
  WeightStencil ws(p.pos, dx, nx, ny);
  float value = 0.f;
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      const int ni = ws.base_i + a;
      const int nj = ws.base_j + b;
      if (!inGrid(ni, nj))
        continue;
      value += phase_laplacian_[gridIdx(ni, nj)] * ws.weight(a, b);
    }
  }
  return value;
}

void IncompressibleBlobTankSimulation::scatterParticleBodyForce(
    const Particle& p, const Eigen::Vector2f& accel) {
  const float dx = params_.dx;
  const int nx = params_.grid_nx;
  const int ny = params_.grid_ny;
  WeightStencil ws(p.pos, dx, nx, ny);
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      const int ni = ws.base_i + a;
      const int nj = ws.base_j + b;
      if (!inGrid(ni, nj))
        continue;
      grid_[gridIdx(ni, nj)].force += ws.weight(a, b) * p.mass * accel;
    }
  }
}

void IncompressibleBlobTankSimulation::substep_addBodyForces() {
  const float g = params_.gravity;

  if (toggles.enable_gravity) {
    for (auto& node : grid_) {
      if (node.mass > 1e-10f)
        node.force.y() += node.mass * g;
    }
  }

  finalizePhaseFields();

  for (const auto& p : particles_) {
    Eigen::Vector2f accel = Eigen::Vector2f::Zero();
    const Eigen::Vector2f grad_phi = samplePhaseGradient(p);
    const float lap_phi = samplePhaseLaplacian(p);
    const float sign = (p.material == MaterialType::Water) ? -1.f : 1.f;

    if (toggles.enable_surface_tension) {
      accel += -scene_params_.surface_tension * params_.dx * params_.dx *
               lap_phi * grad_phi;
    }
    if (toggles.enable_cohesion) {
      accel += scene_params_.cohesion * std::abs(g) * sign * grad_phi;
    }

    if (toggles.enable_stirring) {
      const float tank_mid = 0.5f * (scene_params_.tank_xmin + scene_params_.tank_xmax);
      const float x0 = tank_mid +
                       0.35f * (scene_params_.tank_xmax - scene_params_.tank_xmin) *
                           std::sin(2.f * static_cast<float>(M_PI) *
                                    scene_params_.stirring_frequency * time_);
      const Eigen::Vector2f center(x0, scene_params_.stirring_y);
      const Eigen::Vector2f d = p.pos - center;
      const float r = d.norm();
      if (r < scene_params_.stirring_radius && r > 1e-6f) {
        const float falloff = 1.f - r / scene_params_.stirring_radius;
        const Eigen::Vector2f tangent(-d.y(), d.x());
        accel += scene_params_.stirring_strength * falloff * tangent.normalized();
      }
    }

    if (accel.squaredNorm() > 0.f)
      scatterParticleBodyForce(p, accel);
  }
}

void IncompressibleBlobTankSimulation::substep_project() {
  const int nx = params_.grid_nx;
  const int ny = params_.grid_ny;
  const float dx = params_.dx;
  const float dt = params_.dt;
  const float inv_dx = 1.f / dx;
  const int wall = 2;

  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      auto& node = grid_[gridIdx(i, j)];
      if (node.mass < 1e-10f)
        continue;

      node.vel = node.momentum / node.mass;
      node.vel_new = node.vel + dt * node.force / node.mass;

      if (toggles.no_slip_walls) {
        if (toggles.bc_left && i < wall) {
          node.vel_new.x() = std::max(node.vel_new.x(), 0.f);
          node.vel_new.y() *= 0.5f;
        }
        if (toggles.bc_right && i >= nx - wall) {
          node.vel_new.x() = std::min(node.vel_new.x(), 0.f);
          node.vel_new.y() *= 0.5f;
        }
        if (toggles.bc_bottom && j < wall) {
          node.vel_new.y() = std::max(node.vel_new.y(), 0.f);
          node.vel_new.x() *= 0.5f;
        }
        if (toggles.bc_top && j >= ny - wall) {
          node.vel_new.y() = std::min(node.vel_new.y(), 0.f);
          node.vel_new.x() *= 0.5f;
        }
      }
    }
  }

  if (!toggles.enable_projection)
    return;

  const float rho_ref = 0.5f * (scene_params_.phase_a_density + scene_params_.phase_b_density);
  for (int j = 1; j < ny - 1; ++j) {
    for (int i = 1; i < nx - 1; ++i) {
      const int idx = gridIdx(i, j);
      if (grid_[idx].mass < 1e-10f) {
        divergence_[idx] = 0.f;
        continue;
      }
      const float du = grid_[gridIdx(i + 1, j)].vel_new.x() -
                       grid_[gridIdx(i - 1, j)].vel_new.x();
      const float dv = grid_[gridIdx(i, j + 1)].vel_new.y() -
                       grid_[gridIdx(i, j - 1)].vel_new.y();
      divergence_[idx] = 0.5f * inv_dx * (du + dv);
      pressure_[idx] = 0.f;
    }
  }

  const float alpha = rho_ref * dx * dx / std::max(dt, 1e-8f);
  for (int iter = 0; iter < std::max(scene_params_.pressure_iters, 1); ++iter) {
    for (int j = 1; j < ny - 1; ++j) {
      for (int i = 1; i < nx - 1; ++i) {
        const int idx = gridIdx(i, j);
        if (grid_[idx].mass < 1e-10f) {
          pressure_tmp_[idx] = 0.f;
          continue;
        }
        pressure_tmp_[idx] = 0.25f *
            (pressure_[gridIdx(i - 1, j)] + pressure_[gridIdx(i + 1, j)] +
             pressure_[gridIdx(i, j - 1)] + pressure_[gridIdx(i, j + 1)] -
             alpha * divergence_[idx]);
      }
    }
    std::swap(pressure_, pressure_tmp_);
  }

  for (int j = 1; j < ny - 1; ++j) {
    for (int i = 1; i < nx - 1; ++i) {
      const int idx = gridIdx(i, j);
      auto& node = grid_[idx];
      if (node.mass < 1e-10f)
        continue;
      const float dpdx = 0.5f * inv_dx *
                         (pressure_[gridIdx(i + 1, j)] - pressure_[gridIdx(i - 1, j)]);
      const float dpdy = 0.5f * inv_dx *
                         (pressure_[gridIdx(i, j + 1)] - pressure_[gridIdx(i, j - 1)]);
      node.vel_new -= (dt / rho_ref) * Eigen::Vector2f(dpdx, dpdy);

      if (toggles.bc_left && i < wall) node.vel_new.x() = std::max(node.vel_new.x(), 0.f);
      if (toggles.bc_right && i >= nx - wall) node.vel_new.x() = std::min(node.vel_new.x(), 0.f);
      if (toggles.bc_bottom && j < wall) node.vel_new.y() = std::max(node.vel_new.y(), 0.f);
      if (toggles.bc_top && j >= ny - wall) node.vel_new.y() = std::min(node.vel_new.y(), 0.f);
    }
  }
}

void IncompressibleBlobTankSimulation::substep_G2P() {
  const float dx = params_.dx;
  const float D_inv = params_.D_inv;
  const int nx = params_.grid_nx;
  const int ny = params_.grid_ny;

  for (auto& p : particles_) {
    WeightStencil ws(p.pos, dx, nx, ny);
    Eigen::Vector2f v_new = Eigen::Vector2f::Zero();
    Eigen::Matrix2f C_new = Eigen::Matrix2f::Zero();
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        const int ni = ws.base_i + a;
        const int nj = ws.base_j + b;
        if (!inGrid(ni, nj))
          continue;
        const float w = ws.weight(a, b);
        const Eigen::Vector2f vi = grid_[gridIdx(ni, nj)].vel_new;
        const Eigen::Vector2f xip(Eigen::Vector2f(ni * dx, nj * dx) - p.pos);
        v_new += w * vi;
        C_new += w * (vi * xip.transpose());
      }
    }
    p.vel = v_new;
    p.C = D_inv * C_new;
    p.F.setIdentity();
  }
}

void IncompressibleBlobTankSimulation::enforceTankBounds(Particle& p) const {
  const float xmin = scene_params_.tank_xmin + 0.001f;
  const float xmax = scene_params_.tank_xmax - 0.001f;
  const float ymin = scene_params_.tank_ymin + 0.001f;
  const float ymax = scene_params_.tank_ymax - 0.001f;

  if (toggles.bc_left && p.pos.x() < xmin) {
    p.pos.x() = xmin;
    if (p.vel.x() < 0.f) p.vel.x() = 0.f;
  }
  if (toggles.bc_right && p.pos.x() > xmax) {
    p.pos.x() = xmax;
    if (p.vel.x() > 0.f) p.vel.x() = 0.f;
  }
  if (toggles.bc_bottom && p.pos.y() < ymin) {
    p.pos.y() = ymin;
    if (p.vel.y() < 0.f) p.vel.y() = 0.f;
  }
  if (toggles.bc_top && p.pos.y() > ymax) {
    p.pos.y() = ymax;
    if (p.vel.y() > 0.f) p.vel.y() = 0.f;
  }
}

void IncompressibleBlobTankSimulation::substep_advect() {
  const float dt = params_.dt;
  for (auto& p : particles_) {
    p.pos += dt * p.vel;
    enforceTankBounds(p);
  }
}

void IncompressibleBlobTankSimulation::step() {
  clearGrid();
  substep_P2G();
  substep_addBodyForces();
  substep_project();
  substep_G2P();
  substep_advect();
  ++frame_;
  time_ += params_.dt;
}

void IncompressibleBlobTankSimulation::buildRenderArrays(
    std::vector<std::array<double, 3>>& pos3d,
    std::vector<std::array<double, 3>>& colors) const {
  pos3d.resize(particles_.size());
  colors.resize(particles_.size());
  for (size_t i = 0; i < particles_.size(); ++i) {
    const auto& p = particles_[i];
    pos3d[i] = {static_cast<double>(p.pos.x()), static_cast<double>(p.pos.y()), 0.0};
    const auto& c = (p.material == MaterialType::Water)
                        ? scene_params_.phase_a_color
                        : scene_params_.phase_b_color;
    colors[i] = {c[0], c[1], c[2]};
  }
}

void IncompressibleBlobTankSimulation::registerPolyscope() {
  std::vector<std::array<double, 3>> pos3d, colors;
  buildRenderArrays(pos3d, colors);
  auto* cloud = polyscope::registerPointCloud(kCloudName, pos3d);
  cloud->setPointRadius(0.0035);
  cloud->setPointRenderMode(polyscope::PointRenderMode::Sphere);
  cloud->addColorQuantity("phase_color", colors)->setEnabled(true);
}

void IncompressibleBlobTankSimulation::updatePolyscope(bool force_rebuild) {
  std::vector<std::array<double, 3>> pos3d, colors;
  buildRenderArrays(pos3d, colors);

  auto* cloud = polyscope::hasPointCloud(kCloudName)
                    ? polyscope::getPointCloud(kCloudName)
                    : nullptr;
  if (!cloud || force_rebuild) {
    if (cloud)
      polyscope::removeStructure(kCloudName);
    registerPolyscope();
    return;
  }

  cloud->updatePointPositions(pos3d);
  cloud->addColorQuantity("phase_color", colors)->setEnabled(true);
}
