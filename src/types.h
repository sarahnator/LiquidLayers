#pragma once
#include <Eigen/Dense>
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
//  Particle  — full Phase 2 fields
// ─────────────────────────────────────────────────────────────────────────────
struct Particle {
    // ── Kinematics ────────────────────────────────────────────────────────────
    Eigen::Vector2f pos  = Eigen::Vector2f::Zero();  // world-space position
    Eigen::Vector2f vel  = Eigen::Vector2f::Zero();  // velocity

    // ── APIC affine matrix ────────────────────────────────────────────────────
    // C captures the local velocity gradient around this particle.
    // During G2P: C = sum_i  w_i * v_i * (x_i - x_p)^T * D_p^{-1}
    // During P2G: the contribution to node i is  mass * (vel + C*(x_i - x_p))
    // D_p = (1/4)*dx^2 * I  for quadratic B-splines (precomputed as D_inv below)
    Eigen::Matrix2f C    = Eigen::Matrix2f::Zero();

    // ── Material ──────────────────────────────────────────────────────────────
    MaterialType    material = MaterialType::Water;
    float           mass     = 1.0f;

    // ── Volume / deformation (Phase 3+) ──────────────────────────────────────
    float           vol0     = 0.0f;              // initial volume (set in step 0)
    float           density0 = 1000.0f;           // rest density  (kg/m³)
    Eigen::Matrix2f F        = Eigen::Matrix2f::Identity(); // deformation gradient
};

// ─────────────────────────────────────────────────────────────────────────────
//  Grid node  — lives on the Eulerian background grid
// ─────────────────────────────────────────────────────────────────────────────
struct GridNode {
    float           mass     = 0.0f;
    Eigen::Vector2f momentum = Eigen::Vector2f::Zero(); // = mass * velocity
    Eigen::Vector2f vel      = Eigen::Vector2f::Zero(); // derived: momentum/mass
    Eigen::Vector2f vel_new  = Eigen::Vector2f::Zero(); // after force integration
};

// ─────────────────────────────────────────────────────────────────────────────
//  Quadratic B-spline weight functions
//
//  These are the interpolation kernels that connect particles ↔ grid.
//  We use the quadratic (not cubic) B-spline because:
//    - It's C¹ continuous  → smooth force interpolation
//    - Compact support: affects exactly a 3×3 stencil of grid nodes
//    - Simple to implement and fast to evaluate
//
//  Reference: Jiang et al. 2016 MPM course notes, Appendix B
// ─────────────────────────────────────────────────────────────────────────────

// N(x): 1D quadratic B-spline basis evaluated at normalized distance x = (xp-xi)/dx
// Returns weight in [0, 0.75].
inline float bsplineWeight(float x) {
    x = std::abs(x);
    if (x < 0.5f)        return 0.75f - x * x;
    else if (x < 1.5f) { float t = 1.5f - x; return 0.5f * t * t; }
    else                 return 0.0f;
}

// dN/dx: derivative of the 1D B-spline (needed in Phase 3 for stress → force)
// Returns dw/dx in units of 1/dx  (caller multiplies by 1/dx)
inline float bsplineWeightGrad(float x) {
    float sx = (x >= 0.f) ? 1.f : -1.f;
    x = std::abs(x);
    if (x < 0.5f)       return -2.0f * x * sx;
    else if (x < 1.5f)  return -(1.5f - x) * sx;
    else                return 0.0f;
}

// ─────────────────────────────────────────────────────────────────────────────
//  WeightStencil
//
//  For a particle at position xp, this precomputes the weights and weight
//  gradients for all 9 nodes in its 3×3 stencil.
//
//  Usage:
//    WeightStencil ws(xp, dx, grid_nx, grid_ny);
//    for (int a = 0; a < 3; ++a)
//      for (int b = 0; b < 3; ++b) {
//          int ni = ws.base_i + a,  nj = ws.base_j + b;
//          float w    = ws.w[a]    * ws.w[b];
//          Vec2f dw   = { ws.dw[a] * ws.w[b] / dx,
//                         ws.w[a]  * ws.dw[b] / dx };
//          // ... accumulate P2G or G2P ...
//      }
// ─────────────────────────────────────────────────────────────────────────────
struct WeightStencil {
    int   base_i, base_j;    // grid index of the bottom-left node in the stencil
    float w[3],  wg[3];      // 1D weights and their gradients along X
    float wy[3], wyg[3];     // 1D weights and their gradients along Y

    WeightStencil(const Eigen::Vector2f& xp, float dx, int nx, int ny) {
        // Fractional grid position
        float fx = xp.x() / dx;
        float fy = xp.y() / dx;

        // Base node index: floor(xp/dx - 0.5) for quadratic B-spline
        // The -0.5 shift centers the stencil correctly on the quadratic kernel.
        base_i = static_cast<int>(fx - 0.5f);
        base_j = static_cast<int>(fy - 0.5f);

        // Normalized offsets from base node to particle  (in [0.5, 1.5])
        float ox = fx - static_cast<float>(base_i);   // ∈ [0.5, 1.5]
        float oy = fy - static_cast<float>(base_j);

        // Evaluate 1D weights at offsets 0, 1, 2 from base node
        // offset 0: ox,       offset 1: ox-1,     offset 2: ox-2
        for (int k = 0; k < 3; ++k) {
            float dx_norm = ox - static_cast<float>(k);
            float dy_norm = oy - static_cast<float>(k);
            w[k]   = bsplineWeight(dx_norm);
            wg[k]  = bsplineWeightGrad(dx_norm);
            wy[k]  = bsplineWeight(dy_norm);
            wyg[k] = bsplineWeightGrad(dy_norm);
        }
    }

    // 2D weight for stencil node (a, b)
    float weight(int a, int b) const { return w[a] * wy[b]; }

    // 2D weight gradient for stencil node (a, b), divided by dx
    // (returns ∂w/∂x in world units)
    Eigen::Vector2f weightGrad(int a, int b, float dx) const {
        return { wg[a] * wy[b]  / dx,
                 w[a]  * wyg[b] / dx };
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Simulation parameters
// ─────────────────────────────────────────────────────────────────────────────
struct SimParams {
    // Domain
    float domain_w  = 10.0f;
    float domain_h  =  6.0f;

    // Particle seeding
    int   ppc       = 4;          // particles per cell per axis
    float layer_pct = 0.20f;

    // Grid  — dx = domain_w / grid_nx  (same spacing in both axes)
    int   grid_nx   = 80;
    int   grid_ny   = 48;

    // Time integration
    float dt        = 2e-4f;
    float gravity   = -9.8f;

    // Derived (call computeDerived() after changing grid_nx/domain_w)
    float dx        = 0.0f;       // grid spacing (world units)
    float D_inv     = 0.0f;       // = 4/dx²  — APIC D_p^{-1} scalar

    void computeDerived() {
        dx    = domain_w / static_cast<float>(grid_nx);
        D_inv = 4.0f / (dx * dx);  // for quadratic B-spline: D_p = dx²/4 * I
    }
};
