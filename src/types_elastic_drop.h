#pragma once

#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <vector>

// A deliberately simple single-material 2D MPM setup for debugging gravity +
// linear elasticity in a box. The constitutive law is the small-strain isotropic
// linear elastic model
//
//   sigma = lambda tr(eps) I + 2 mu eps,
//   eps   = sym(grad u) = 0.5 * ((F - I) + (F - I)^T).
//
// In a 2D code there are two common reductions of the 3D isotropic law:
// plane strain and plane stress. We expose both, but default to plane strain
// because it is a common choice for 2D MPM debugging.

enum class MaterialType : int {
  Elastic = 0,
};

inline std::array<float, 3> materialColor(MaterialType) {
  return {0.82f, 0.28f, 0.28f};
}

enum class ConstitutiveModel {
  LinearElasticSmallStrain,
};

struct MaterialParams {
  ConstitutiveModel model = ConstitutiveModel::LinearElasticSmallStrain;
  float density0 = 1000.f;
  float youngs_modulus = 2.0e4f;
  float poisson_ratio = 0.30f;
  bool plane_stress = false; // false = plane strain

  // Derived Lamé parameters.
  float mu = 0.f;
  float lambda_lame = 0.f;

  void computeDerived() {
    const float nu = poisson_ratio;
    const float E = youngs_modulus;
    mu = E / (2.f * (1.f + nu));
    lambda_lame = E * nu / ((1.f + nu) * (1.f - 2.f * nu));
  }

  // Effective 2D lambda for the chosen reduction.
  float lambda2D() const {
    if (!plane_stress)
      return lambda_lame; // plane strain

    // Plane stress reduction:
    // sigma = 2 mu eps + lambda_ps tr(eps) I,
    // lambda_ps = E nu / (1 - nu^2) = 2 mu nu / (1 - nu)
    const float nu = poisson_ratio;
    return (2.f * mu * nu) / std::max(1.f - nu, 1e-6f);
  }
};

struct Particle {
  Eigen::Vector2f pos = Eigen::Vector2f::Zero();
  Eigen::Vector2f vel = Eigen::Vector2f::Zero();
  Eigen::Matrix2f C = Eigen::Matrix2f::Zero();
  Eigen::Matrix2f F = Eigen::Matrix2f::Identity();

  MaterialType material = MaterialType::Elastic;
  float mass = 1.f;
  float vol0 = 0.f;
};

struct GridNode {
  float mass = 0.f;
  Eigen::Vector2f momentum = Eigen::Vector2f::Zero();
  Eigen::Vector2f vel = Eigen::Vector2f::Zero();
  Eigen::Vector2f vel_new = Eigen::Vector2f::Zero();
  Eigen::Vector2f force = Eigen::Vector2f::Zero();
};

inline float bsplineWeight(float x) {
  x = std::abs(x);
  if (x < 0.5f)
    return 0.75f - x * x;
  if (x < 1.5f) {
    const float t = 1.5f - x;
    return 0.5f * t * t;
  }
  return 0.f;
}

inline float bsplineWeightGrad(float x) {
  const float sx = (x >= 0.f) ? 1.f : -1.f;
  x = std::abs(x);
  if (x < 0.5f)
    return -2.f * x * sx;
  if (x < 1.5f)
    return -(1.5f - x) * sx;
  return 0.f;
}

struct WeightStencil {
  int base_i = 0;
  int base_j = 0;
  float w[3]{};
  float wg[3]{};
  float wy[3]{};
  float wyg[3]{};

  WeightStencil(const Eigen::Vector2f &xp, float dx, int, int) {
    const float fx = xp.x() / dx;
    const float fy = xp.y() / dx;
    base_i = static_cast<int>(fx - 0.5f);
    base_j = static_cast<int>(fy - 0.5f);
    const float ox = fx - static_cast<float>(base_i);
    const float oy = fy - static_cast<float>(base_j);
    for (int k = 0; k < 3; ++k) {
      w[k] = bsplineWeight(ox - static_cast<float>(k));
      wg[k] = bsplineWeightGrad(ox - static_cast<float>(k));
      wy[k] = bsplineWeight(oy - static_cast<float>(k));
      wyg[k] = bsplineWeightGrad(oy - static_cast<float>(k));
    }
  }

  float weight(int a, int b) const { return w[a] * wy[b]; }

  Eigen::Vector2f weightGrad(int a, int b, float dx) const {
    return {wg[a] * wy[b] / dx, w[a] * wyg[b] / dx};
  }
};

struct SimParams {
  float domain_w = 10.f;
  float domain_h = 6.f;
  int ppc = 4;
  int grid_nx = 80;
  int grid_ny = 48;
  float dt = 5e-5f;
  float gravity = -9.8f;

  // Initial drop configuration.
  float drop_w = 1.2f;
  float drop_h = 1.2f;
  float drop_center_x = 5.f;
  float drop_center_y = 4.5f;

  // Derived values.
  float dx = 0.f;
  float D_inv = 0.f;

  MaterialParams elastic_material{};

  void computeDerived() {
    dx = domain_w / static_cast<float>(grid_nx);
    D_inv = 4.f / (dx * dx);
    elastic_material.computeDerived();
  }
};
