#include "simulation_elastic_drop.h"
#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include <algorithm>
#include <cmath>
#include <iostream>

Simulation::Simulation(SimParams params) : params_(std::move(params)) {
  params_.computeDerived();
  grid_.resize(static_cast<size_t>(params_.grid_nx * params_.grid_ny));
}

void Simulation::addElasticBlock(float x_min, float x_max, float y_min,
                                 float y_max) {
  const float dx = params_.dx;
  const int ppc = params_.ppc;
  const float px = dx / static_cast<float>(ppc);
  const float py = dx / static_cast<float>(ppc);

  auto jitter = [&](int i, float s) -> float {
    return s *
           (static_cast<float>((i * 1013904223u + 1664525u) & 0xFFFFu) /
                65535.f -
            0.5f);
  };

  const MaterialParams &mp = params_.elastic_material;
  int pidx = 0;
  for (float y = y_min + 0.5f * py; y < y_max; y += py) {
    for (float x = x_min + 0.5f * px; x < x_max; x += px) {
      Particle p;
      p.pos.x() = std::clamp(x + jitter(2 * pidx, 0.2f * px), 0.001f,
                             params_.domain_w - 0.001f);
      p.pos.y() = std::clamp(y + jitter(2 * pidx + 1, 0.2f * py), 0.001f,
                             params_.domain_h - 0.001f);
      p.material = MaterialType::Elastic;
      p.mass = mp.density0 * px * py;
      p.vol0 = px * py;
      p.F = Eigen::Matrix2f::Identity();
      particles_.push_back(p);
      ++pidx;
    }
  }
}

void Simulation::initialize() {
  particles_.clear();
  frame_ = 0;

  const float hw = 0.5f * params_.drop_w;
  const float hh = 0.5f * params_.drop_h;
  addElasticBlock(params_.drop_center_x - hw, params_.drop_center_x + hw,
                  params_.drop_center_y - hh, params_.drop_center_y + hh);

  std::cout << "[Elastic MPM debug] initialized " << particles_.size()
            << " particles\n";
}

Eigen::Matrix2f Simulation::stressLinearElastic(const Particle &p,
                                                const MaterialParams &mp) const {
  // Small-strain law based directly on the generalized Hooke law.
  // Let grad u = F - I. Then
  //   eps = sym(grad u) = 0.5 * ((F - I) + (F - I)^T).
  // For isotropic elasticity in 2D,
  //   sigma = lambda_2D tr(eps) I + 2 mu eps.
  // The grid force update in this MPM code uses Kirchhoff stress tau, where
  //   tau = J sigma,  J = det(F).
  const Eigen::Matrix2f I = Eigen::Matrix2f::Identity();
  const Eigen::Matrix2f grad_u = p.F - I;
  const Eigen::Matrix2f eps = 0.5f * (grad_u + grad_u.transpose());

  const float lambda_2d = mp.lambda2D();
  Eigen::Matrix2f sigma = 2.f * mp.mu * eps +
                          lambda_2d * eps.trace() * Eigen::Matrix2f::Identity();

  const float J = std::clamp(p.F.determinant(), 0.2f, 5.f);
  return J * sigma;
}

Eigen::Matrix2f Simulation::kirchhoffStress(const Particle &p) const {
  return stressLinearElastic(p, params_.elastic_material);
}

void Simulation::clearGrid() {
  for (auto &n : grid_) {
    n.mass = 0.f;
    n.momentum = Eigen::Vector2f::Zero();
    n.vel = Eigen::Vector2f::Zero();
    n.vel_new = Eigen::Vector2f::Zero();
    n.force = Eigen::Vector2f::Zero();
  }
}

void Simulation::substep_P2G() {
  const float dx = params_.dx;
  const int nx = params_.grid_nx;
  const int ny = params_.grid_ny;

  for (const auto &p : particles_) {
    WeightStencil ws(p.pos, dx, nx, ny);
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        const int ni = ws.base_i + a;
        const int nj = ws.base_j + b;
        if (!inGrid(ni, nj))
          continue;

        const float w = ws.weight(a, b);
        const Eigen::Vector2f xip = Eigen::Vector2f(ni * dx, nj * dx) - p.pos;
        auto &node = grid_[gridIdx(ni, nj)];
        node.mass += w * p.mass;
        node.momentum += w * p.mass * (p.vel + p.C * xip);
      }
    }
  }
}

void Simulation::substep_gridUpdate() {
  const float dx = params_.dx;
  const float dt = params_.dt;
  const int nx = params_.grid_nx;
  const int ny = params_.grid_ny;
  const int wall = 2;

  if (toggles.enable_stress) {
    for (const auto &p : particles_) {
      const Eigen::Matrix2f tau = kirchhoffStress(p);
      WeightStencil ws(p.pos, dx, nx, ny);
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          const int ni = ws.base_i + a;
          const int nj = ws.base_j + b;
          if (!inGrid(ni, nj))
            continue;
          grid_[gridIdx(ni, nj)].force -= p.vol0 * (tau * ws.weightGrad(a, b, dx));
        }
      }
    }
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

      if (toggles.sticky_walls) {
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
}

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
    }

    p.vel = v_new;
    p.C = D_inv * C_new;
    p.F = (Eigen::Matrix2f::Identity() + dt * p.C) * p.F;
  }
}

void Simulation::enforceParticleBounds(Particle &p) const {
  const float eps = 1e-3f;
  const float W = params_.domain_w;
  const float H = params_.domain_h;

  if (toggles.bc_left && p.pos.x() < eps) {
    p.pos.x() = eps;
    if (p.vel.x() < 0.f)
      p.vel.x() = 0.f;
  }
  if (toggles.bc_right && p.pos.x() > W - eps) {
    p.pos.x() = W - eps;
    if (p.vel.x() > 0.f)
      p.vel.x() = 0.f;
  }
  if (toggles.bc_bottom && p.pos.y() < eps) {
    p.pos.y() = eps;
    if (p.vel.y() < 0.f)
      p.vel.y() = 0.f;
  }
  if (toggles.bc_top && p.pos.y() > H - eps) {
    p.pos.y() = H - eps;
    if (p.vel.y() > 0.f)
      p.vel.y() = 0.f;
  }
}

void Simulation::substep_advect() {
  const float dt = params_.dt;
  for (auto &p : particles_) {
    p.pos += dt * p.vel;
    enforceParticleBounds(p);
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

void Simulation::buildRenderArrays(
    std::vector<std::array<double, 3>> &pos3d,
    std::vector<std::array<double, 3>> &colors) const {
  pos3d.resize(particles_.size());
  colors.resize(particles_.size());
  for (size_t i = 0; i < particles_.size(); ++i) {
    const auto &p = particles_[i];
    pos3d[i] = {static_cast<double>(p.pos.x()), static_cast<double>(p.pos.y()), 0.0};
    const auto c = materialColor(p.material);
    colors[i] = {c[0], c[1], c[2]};
  }
}

void Simulation::registerPolyscope() {
  std::vector<std::array<double, 3>> pos3d, colors;
  buildRenderArrays(pos3d, colors);
  auto *cloud = polyscope::registerPointCloud(kCloudName, pos3d);
  cloud->setPointRadius(0.0035);
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
