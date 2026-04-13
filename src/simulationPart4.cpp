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
  std::vector<Layer> layers = {
      {MaterialType::Rock, 0.f * H * pct, 1.f * H * pct},
      {MaterialType::Sand, 1.f * H * pct, 2.f * H * pct},
      {MaterialType::Soil, 2.f * H * pct, 3.f * H * pct},
      {MaterialType::Water, 3.f * H * pct, 4.f * H * pct},
  };
  const float px = dx / ppc, py = dx / ppc;
  auto jitter = [&](int i, float s) -> float {
    return s *
           (static_cast<float>((i * 1013904223 + 1664525) & 0xFFFF) / 65535.f -
            0.5f);
  };
  int pidx = 0;
  for (const auto &l : layers) {
    MaterialParams mp = defaultMaterialParams(l.mat);
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
  std::cout << "[MPM P4] " << particles_.size() << " particles\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  stressFluid()
//
//  Weakly compressible fluid with F-reset each step.
//
//  The F-reset trick:  for a fluid, F accumulates large shear errors over time
//  because the particle neighbourhood changes topology.  Instead of integrating
//  the full F, we:
//    1.  Extract J = det(F)  (volume change — the physically meaningful part)
//    2.  Clamp J to a safe range to prevent pressure blow-up
//    3.  Compute pressure from J via the Tait EOS
//    4.  Reset F to sqrt(J)*I so only volume info is preserved for next step
//
//  This is the approach used by Grant Kot and described in niall's MPM guide.
//  It allows a larger dt than keeping F intact.
// ─────────────────────────────────────────────────────────────────────────────
Eigen::Matrix2f Simulation::stressFluid(const Particle &p,
                                        const MaterialParams &mp) const {
  float J = p.F.determinant();
  J = std::clamp(J, 0.6f, 1.4f); // tight clamp — prevents EOS blow-up

  // Tait equation of state:  pressure = k * (J^{-gamma} - 1)
  float pressure = mp.bulk_modulus * (std::exp(-mp.gamma * std::log(J)) - 1.f);

  // Kirchhoff stress:  tau = -J*p*I + J*mu*(C + C^T)   (viscous term)
  Eigen::Matrix2f tau = -J * pressure * Eigen::Matrix2f::Identity();
  if (mp.viscosity > 0.f)
    tau += J * mp.viscosity * (p.C + p.C.transpose());

  return tau;
}

// ─────────────────────────────────────────────────────────────────────────────
//  stressFixedCorotated()
//
//  Elastic solid stress using the fixed-corotated model (Stomakhin et al.
//  2012).
//
//  The energy density is:
//    Psi(F) = mu * ||F - R||_F^2  +  (lambda/2) * (J-1)^2
//
//  The first term is the Neo-Hookean shear energy — it penalises deviation
//  from a pure rotation.  The second term is volumetric energy that penalises
//  compression or expansion away from J=1.
//
//  The Kirchhoff stress is  tau = dPsi/dF * F^T:
//    tau = 2*mu*(F - R)*F^T  +  lambda*(J-1)*J*I
//
//  Derivation sketch:
//    d/dF [||F-R||_F^2] = 2*(F-R)           (R treated as fixed wrt F)
//    d/dF [(J-1)^2/2]  = (J-1)*dJ/dF = (J-1)*J*F^{-T}
//    tau = (dPsi/dF)*F^T = 2mu*(F-R)*F^T + lambda*(J-1)*J*F^{-T}*F^T
//        = 2mu*(F-R)*F^T + lambda*(J-1)*J*I
//
//  R comes from polar decomposition F = R*S.  We compute it via SVD.
// ─────────────────────────────────────────────────────────────────────────────
Eigen::Matrix2f
Simulation::stressFixedCorotated(const Particle &p,
                                 const MaterialParams &mp) const {
  Eigen::Matrix2f R, S;
  polarDecompose2x2(p.F, R, S);

  float J = p.F.determinant();
  J = std::clamp(J, 0.2f, 5.f);

  // tau = 2*mu*(F - R)*F^T + lambda*(J-1)*J*I
  Eigen::Matrix2f tau =
      2.f * mp.mu * (p.F - R) * p.F.transpose() +
      mp.lambda_lame * (J - 1.f) * J * Eigen::Matrix2f::Identity();

  return tau;
}

// ─────────────────────────────────────────────────────────────────────────────
//  stressDruckerPrager()
//
//  Elastoplastic stress for sand.  The elastic part F^E is stored in p.F
//  (it has already been projected to the yield surface at the end of G2P via
//  projectDruckerPrager).  We compute stress from F^E exactly as if it were
//  a fixed-corotated elastic material — the plasticity is handled separately
//  in the projection step, not here.
//
//  This separation is the key idea of the return mapping algorithm:
//    - Treat F^E as elastic during stress computation
//    - Project F^E to the yield surface during the plasticity update
//
//  For sand we use only the volumetric part of the fixed-corotated model
//  (no shear resistance once yielded), but the full model is applied
//  elastically:
//    tau = 2*mu*(F^E - R^E)*F^{E,T} + lambda*(J^E-1)*J^E*I
// ─────────────────────────────────────────────────────────────────────────────
Eigen::Matrix2f
Simulation::stressDruckerPrager(const Particle &p,
                                const MaterialParams &mp) const {
  // Stress computation is identical to fixed-corotated — plasticity was
  // handled when projectDruckerPrager() modified p.F at the end of G2P.
  return stressFixedCorotated(p, mp);
}

// ─────────────────────────────────────────────────────────────────────────────
//  kirchhoffStress()  — dispatcher
// ─────────────────────────────────────────────────────────────────────────────
Eigen::Matrix2f Simulation::kirchhoffStress(const Particle &p) const {
  MaterialParams mp = defaultMaterialParams(p.material);
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

// ─────────────────────────────────────────────────────────────────────────────
//  projectDruckerPrager()
//
//  Return mapping for the Drucker-Prager yield criterion in 2D.
//  Projects the elastic trial state F^E back to the yield surface if violated.
//
//  Algorithm (Klar et al. 2016, "Drucker-Prager Elastoplasticity for Sand
//  Animation", SIGGRAPH 2016):
//
//  1.  SVD:  F^E = U * Sigma * V^T
//      Singular values Sigma = diag(s1, s2) give principal stretches.
//
//  2.  Hencky (log) strains:  epsilon = log(Sigma) = (log s1, log s2)
//      These are the "natural" strain measure for large-deformation plasticity.
//      At zero deformation, Sigma=I so epsilon=0.
//
//  3.  Decompose epsilon into:
//      - deviatoric part:  epsilon_dev = epsilon - (tr/2)*I   (shape change)
//      - volumetric part:  tr(epsilon)                        (volume change)
//
//  4.  Drucker-Prager yield function:
//      f = ||epsilon_dev|| + alpha * tr(epsilon)
//      Yield if f > 0 (or tr >= 0 for pure tension).
//
//  5.  Three cases:
//      (a) tr(epsilon) >= 0 (tension):  set epsilon = 0  (sand can't be pulled)
//      (b) f <= 0 (within yield cone):  no projection needed (elastic)
//      (c) otherwise (compressive shear exceeds cone):
//          project epsilon_dev onto the cone boundary:
//            epsilon_new = epsilon - (f/||epsilon_dev||) * epsilon_dev
//
//  6.  Reconstruct F^E from projected epsilon:
//      F^E_new = U * exp(epsilon_new) * V^T
//
//  Why SVD / log-strain space?  The yield criterion and return mapping have
//  elegant closed forms when expressed in terms of singular values.  This
//  approach avoids the need to solve a nonlinear system for the plastic
//  multiplier, which is required for stress-space formulations.
// ─────────────────────────────────────────────────────────────────────────────
void Simulation::projectDruckerPrager(Particle &p,
                                      const MaterialParams &mp) const {
  // Step 1: SVD of elastic F
  Eigen::Matrix2f U, V;
  Eigen::Vector2f sigma_vec;
  svd2x2(p.F, U, sigma_vec, V);

  // Clamp singular values to prevent log(0) or negative logs
  sigma_vec(0) = std::max(sigma_vec(0), 0.05f);
  sigma_vec(1) = std::max(sigma_vec(1), 0.05f);

  // Step 2: Hencky strains
  Eigen::Vector2f eps(std::log(sigma_vec(0)), std::log(sigma_vec(1)));

  // Step 3: decompose into volumetric and deviatoric parts
  float tr_eps = eps(0) + eps(1); // tr(epsilon)
  Eigen::Vector2f eps_dev = eps - (tr_eps / 2.f) * Eigen::Vector2f::Ones();
  float dev_norm = eps_dev.norm(); // ||epsilon_dev||

  // Step 4/5: check yield and project
  bool in_tension = (tr_eps >= 0.f);
  float yield_value = dev_norm + mp.alpha_dp * tr_eps;
  bool elastic_regime = (!in_tension && yield_value <= 0.f);

  if (elastic_regime) {
    // No plasticity — F^E is unchanged
    return;
  }

  Eigen::Vector2f eps_new;

  if (in_tension || dev_norm < 1e-10f) {
    // Case (a): tension or zero deviatoric strain — collapse to zero
    // Sand can't sustain tensile stress at all.  Reset F^E to identity.
    eps_new = Eigen::Vector2f::Zero();
  } else {
    // Case (c): project deviatoric component onto cone surface
    // The cone surface is at:  ||eps_dev|| = -alpha * tr(eps)
    // So we scale eps_dev to have that magnitude, keeping direction.
    float scale = -mp.alpha_dp * tr_eps / dev_norm;
    eps_new = scale * eps_dev + (tr_eps / 2.f) * Eigen::Vector2f::Ones();
  }

  // Step 6: reconstruct F^E from projected epsilon
  // exp(epsilon) gives new singular values, then F^E = U * diag * V^T
  Eigen::Vector2f sigma_new(std::exp(eps_new(0)), std::exp(eps_new(1)));
  p.F = U * sigma_new.asDiagonal() * V.transpose();
}

// ─────────────────────────────────────────────────────────────────────────────
//  clearGrid()
// ─────────────────────────────────────────────────────────────────────────────
void Simulation::clearGrid() {
  for (auto &n : grid_) {
    n.mass = 0.f;
    n.momentum = Eigen::Vector2f::Zero();
    n.vel = n.vel_new = n.force = Eigen::Vector2f::Zero();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  substep_P2G()  — unchanged from Phase 3
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
//  substep_gridUpdate()  — Phase 4: same structure as Phase 3
// ─────────────────────────────────────────────────────────────────────────────
void Simulation::substep_gridUpdate() {
  const float dx = params_.dx, dt = params_.dt;
  const int nx = params_.grid_nx, ny = params_.grid_ny, wall = 2;

  // Pass 1: scatter stress forces
  for (const auto &p : particles_) {
    Eigen::Matrix2f tau = kirchhoffStress(p);
    WeightStencil ws(p.pos, dx, nx, ny);
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) {
        int ni = ws.base_i + a, nj = ws.base_j + b;
        if (!inGrid(ni, nj))
          continue;
        Eigen::Vector2f grad_w = ws.weightGrad(a, b, dx);
        grid_[gridIdx(ni, nj)].force -= p.vol0 * (tau * grad_w);
      }
  }

  // Pass 2: velocity update + gravity + BCs
  for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
      auto &node = grid_[gridIdx(i, j)];
      if (node.mass < 1e-10f)
        continue;
      node.vel = node.momentum / node.mass;
      node.force.y() += node.mass * params_.gravity;
      node.vel_new = node.vel + dt * node.force / node.mass;
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

// ─────────────────────────────────────────────────────────────────────────────
//  substep_G2P()
//
//  Phase 4 addition: after updating vel, C, and F for each particle,
//  apply plasticity projection for sand particles.
//
//  Also apply F-reset for fluid particles: after extracting J, reset F
//  to preserve only volume information, discarding accumulated shear.
//  This is what prevents the frame-230 blow-up.
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

    // Update F (trial elastic deformation gradient)
    p.F = (Eigen::Matrix2f::Identity() + dt * p.C) * p.F;

    MaterialParams mp = defaultMaterialParams(p.material);

    if (mp.model == ConstitutiveModel::WeaklyCompressibleFluid) {
      // ── F-reset for fluids ────────────────────────────────────────────
      // Extract J, clamp, then reset F = sqrt(J)*I.
      // This discards accumulated shear errors while preserving the
      // volume change that drives the pressure.
      float J = std::clamp(p.F.determinant(), 0.6f, 1.4f);
      p.F = std::sqrt(J) * Eigen::Matrix2f::Identity();

    } else if (mp.model == ConstitutiveModel::DruckerPrager) {
      // ── Drucker-Prager return mapping for sand ────────────────────────
      // Projects F^E onto the yield surface if the trial state violates
      // the Drucker-Prager yield criterion.
      projectDruckerPrager(p, mp);
    }
    // FixedCorotated (rock, soil): no plastic projection needed —
    // F is kept as-is and drives the elastic stress directly.
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
