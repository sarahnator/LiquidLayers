#pragma once
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <vector>
#include <array>
#include <cmath>

// ─────────────────────────────────────────────────────────────────────────────
//  Material types
// ─────────────────────────────────────────────────────────────────────────────
enum class MaterialType : int {
    Water = 0,
    Soil  = 1,
    Sand  = 2,
    Rock  = 3,
};

inline std::array<float,3> materialColor(MaterialType m) {
    switch (m) {
        case MaterialType::Water: return {0.22f, 0.47f, 0.82f};
        case MaterialType::Soil:  return {0.20f, 0.60f, 0.25f};
        case MaterialType::Sand:  return {0.85f, 0.60f, 0.15f};
        case MaterialType::Rock:  return {0.72f, 0.18f, 0.18f};
    }
    return {1,1,1};
}

// ─────────────────────────────────────────────────────────────────────────────
//  MaterialParams — Phase 4 extended with elasticity and plasticity params
// ─────────────────────────────────────────────────────────────────────────────
enum class ConstitutiveModel {
    WeaklyCompressibleFluid,   // Water: Tait EOS + F-reset
    FixedCorotated,            // Rock, Soil: elastic solid, polar decomp stress
    DruckerPrager,             // Sand: elastoplastic with friction cone
};

struct MaterialParams {
    ConstitutiveModel model   = ConstitutiveModel::WeaklyCompressibleFluid;
    float density0            = 1000.f;

    // ── Fluid params (WeaklyCompressibleFluid) ────────────────────────────────
    float bulk_modulus        = 500.f;
    float gamma               = 7.f;
    float viscosity           = 0.05f;

    // ── Elastic params (FixedCorotated, DruckerPrager) ────────────────────────
    // Lame coefficients derived from Young's modulus E and Poisson's ratio nu:
    //   mu     = E / (2*(1+nu))      shear modulus
    //   lambda = E*nu / ((1+nu)*(1-2*nu))  first Lame parameter (bulk-like)
    float youngs_modulus      = 1e4f;
    float poisson_ratio       = 0.3f;

    // Derived Lame coefficients — call computeLame() after setting E and nu
    float mu                  = 0.f;
    float lambda_lame         = 0.f;

    // ── Drucker-Prager plasticity params (Sand) ───────────────────────────────
    // friction_angle: internal friction angle phi (degrees).
    //   Typical sand: 30-40 degrees.  Higher = steeper yield cone = less plastic.
    // cohesion: yield stress at zero pressure (allows some tensile strength).
    //   Pure sand: 0.  Slightly cohesive soil: small positive value.
    float friction_angle      = 30.f;   // degrees
    float cohesion            = 0.f;

    // Derived Drucker-Prager parameter alpha  (computed from friction_angle)
    //   alpha = 2*sin(phi) / (sqrt(3)*(3-sin(phi)))
    //   Controls how strongly pressure affects the yield stress.
    float alpha_dp            = 0.f;    // computed in computeDerived()

    void computeDerived() {
        float nu  = poisson_ratio;
        float E   = youngs_modulus;
        mu           = E / (2.f * (1.f + nu));
        lambda_lame  = E * nu / ((1.f + nu) * (1.f - 2.f * nu));

        float phi_rad = friction_angle * (float)M_PI / 180.f;
        float sp = std::sin(phi_rad);
        alpha_dp = 2.f * sp / (std::sqrt(3.f) * (3.f - sp));
    }
};

inline MaterialParams defaultMaterialParams(MaterialType m) {
    MaterialParams p;
    switch (m) {
        case MaterialType::Water:
            p.model          = ConstitutiveModel::WeaklyCompressibleFluid;
            p.density0       = 1000.f;
            p.bulk_modulus   = 500.f;
            p.gamma          = 7.f;
            p.viscosity      = 0.05f;
            break;

        case MaterialType::Soil:
            // Soil: softer elastic solid — resists compression and shear,
            // but yields at lower stress than rock.
            p.model          = ConstitutiveModel::FixedCorotated;
            p.density0       = 1300.f;
            p.youngs_modulus = 8e3f;
            p.poisson_ratio  = 0.35f;
            break;

        case MaterialType::Sand:
            // Sand: elastoplastic with Drucker-Prager friction cone.
            // Elastic until yield, then flows without tensile stress.
            p.model          = ConstitutiveModel::DruckerPrager;
            p.density0       = 1600.f;
            p.youngs_modulus = 1.5e4f;
            p.poisson_ratio  = 0.30f;
            p.friction_angle = 35.f;
            p.cohesion       = 0.f;
            break;

        case MaterialType::Rock:
            // Rock: stiff elastic solid.  High modulus, low Poisson ratio.
            p.model          = ConstitutiveModel::FixedCorotated;
            p.density0       = 2500.f;
            p.youngs_modulus = 5e4f;
            p.poisson_ratio  = 0.25f;
            break;
    }
    p.computeDerived();
    return p;
}

// ─────────────────────────────────────────────────────────────────────────────
//  2D SVD helper
//
//  Computes the thin SVD of a 2x2 matrix A = U * Sigma * V^T.
//  Returns U, singular values as a Vector2f, and V.
//  U and V are rotation matrices (det = +1) — we enforce this by flipping
//  signs so that Sigma entries are non-negative.  This is important for
//  the polar decomposition and the Drucker-Prager return mapping.
// ─────────────────────────────────────────────────────────────────────────────
inline void svd2x2(
    const Eigen::Matrix2f& A,
    Eigen::Matrix2f& U,
    Eigen::Vector2f& sigma,
    Eigen::Matrix2f& V)
{
    Eigen::JacobiSVD<Eigen::Matrix2f> svd(
        A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    U     = svd.matrixU();
    V     = svd.matrixV();
    sigma = svd.singularValues();

    // Enforce det(U) = det(V) = +1 (proper rotation, not reflection).
    // JacobiSVD may return det = -1; fix by negating one column and
    // the corresponding singular value.
    if (U.determinant() < 0.f) {
        U.col(0) *= -1.f;
        sigma(0) *= -1.f;
    }
    if (V.determinant() < 0.f) {
        V.col(0) *= -1.f;
        sigma(0) *= -1.f;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Polar decomposition  R, S  from  F = R*S
//  Uses SVD:  F = U*Sigma*V^T  =>  R = U*V^T,  S = V*Sigma*V^T
// ─────────────────────────────────────────────────────────────────────────────
inline void polarDecompose2x2(
    const Eigen::Matrix2f& F,
    Eigen::Matrix2f& R,
    Eigen::Matrix2f& S)
{
    Eigen::Matrix2f U, V;
    Eigen::Vector2f sigma;
    svd2x2(F, U, sigma, V);
    R = U * V.transpose();
    S = V * sigma.asDiagonal() * V.transpose();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Particle — Phase 4: separate F_elastic for plasticity
// ─────────────────────────────────────────────────────────────────────────────
struct Particle {
    Eigen::Vector2f pos      = Eigen::Vector2f::Zero();
    Eigen::Vector2f vel      = Eigen::Vector2f::Zero();
    Eigen::Matrix2f C        = Eigen::Matrix2f::Zero();

    // For elastic materials: F is the total deformation gradient.
    // For plastic materials (sand): F is the ELASTIC part F^E.
    //   The plastic part F^P is implicit — it's encoded in the fact that
    //   we project F^E back to the yield surface at the end of each step.
    Eigen::Matrix2f F        = Eigen::Matrix2f::Identity();

    MaterialType    material = MaterialType::Water;
    float           mass     = 1.f;
    float           vol0     = 0.f;
};

// ─────────────────────────────────────────────────────────────────────────────
//  GridNode
// ─────────────────────────────────────────────────────────────────────────────
struct GridNode {
    float           mass     = 0.f;
    Eigen::Vector2f momentum = Eigen::Vector2f::Zero();
    Eigen::Vector2f vel      = Eigen::Vector2f::Zero();
    Eigen::Vector2f vel_new  = Eigen::Vector2f::Zero();
    Eigen::Vector2f force    = Eigen::Vector2f::Zero();
};

// ─────────────────────────────────────────────────────────────────────────────
//  B-spline weights
// ─────────────────────────────────────────────────────────────────────────────
inline float bsplineWeight(float x) {
    x = std::abs(x);
    if (x < 0.5f)       return 0.75f - x*x;
    else if (x < 1.5f) { float t = 1.5f - x; return 0.5f*t*t; }
    else                return 0.f;
}
inline float bsplineWeightGrad(float x) {
    float sx = (x >= 0.f) ? 1.f : -1.f;
    x = std::abs(x);
    if (x < 0.5f)       return -2.f*x*sx;
    else if (x < 1.5f)  return -(1.5f-x)*sx;
    else                return 0.f;
}

struct WeightStencil {
    int   base_i, base_j;
    float w[3], wg[3], wy[3], wyg[3];

    WeightStencil(const Eigen::Vector2f& xp, float dx, int nx, int ny) {
        float fx = xp.x()/dx, fy = xp.y()/dx;
        base_i = static_cast<int>(fx - 0.5f);
        base_j = static_cast<int>(fy - 0.5f);
        float ox = fx - base_i, oy = fy - base_j;
        for (int k = 0; k < 3; ++k) {
            w[k]   = bsplineWeight(ox - k);
            wg[k]  = bsplineWeightGrad(ox - k);
            wy[k]  = bsplineWeight(oy - k);
            wyg[k] = bsplineWeightGrad(oy - k);
        }
    }
    float weight(int a, int b) const { return w[a]*wy[b]; }
    Eigen::Vector2f weightGrad(int a, int b, float dx) const {
        return { wg[a]*wy[b]/dx, w[a]*wyg[b]/dx };
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  SimParams
// ─────────────────────────────────────────────────────────────────────────────
struct SimParams {
    float domain_w  = 10.f;
    float domain_h  =  6.f;
    int   ppc       = 4;
    float layer_pct = 0.20f;
    int   grid_nx   = 80;
    int   grid_ny   = 48;
    float dt        = 5e-5f;   // smaller for stiffer elastic materials
    float gravity   = -9.8f;
    float dx        = 0.f;
    float D_inv     = 0.f;
    void computeDerived() {
        dx    = domain_w / grid_nx;
        D_inv = 4.f / (dx*dx);
    }
};
