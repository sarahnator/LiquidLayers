#pragma once
#include <Eigen/Dense>
#include <vector>
#include <array>

// ─────────────────────────────────────────────────────────────────────────────
//  Material types  (one per colored layer, like Grant Kot's LL)
// ─────────────────────────────────────────────────────────────────────────────
enum class MaterialType : int {
    Water  = 0,   // blue   — top layer
    Soil   = 1,   // green
    Sand   = 2,   // orange
    Rock   = 3,   // red    — bottom layer
};

// RGBA colors matching the screenshot
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
//  Particle  — the fundamental MPM Lagrangian entity
//
//  In Phase 1 (today) we only use: position, color.
//  Phase 2-3 will add: velocity, mass, volume, deformation gradient F, stress.
//  They're commented-out here so you can see where they'll go.
// ─────────────────────────────────────────────────────────────────────────────
struct Particle {
    Eigen::Vector2f pos;          // world-space position (x, y)
    MaterialType    material;

    // ── Phase 2+ fields (uncomment when you reach those phases) ──────────────
    // Eigen::Vector2f vel      = Eigen::Vector2f::Zero();  // velocity
    // Eigen::Matrix2f C        = Eigen::Matrix2f::Zero();  // APIC affine matrix
    // float           mass     = 1.0f;
    // float           vol0     = 0.0f;                     // initial volume
    // Eigen::Matrix2f F        = Eigen::Matrix2f::Identity(); // deformation gradient
};

// ─────────────────────────────────────────────────────────────────────────────
//  Grid cell  — the Eulerian background grid node
//
//  Phase 1: not used yet (particles are just rendered as points).
//  Phase 2-3: we will rasterize particle mass/momentum here, then integrate.
// ─────────────────────────────────────────────────────────────────────────────
struct GridCell {
    // Eigen::Vector2f vel      = Eigen::Vector2f::Zero();
    // Eigen::Vector2f vel_new  = Eigen::Vector2f::Zero();
    // float           mass     = 0.0f;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Simulation parameters (tweak these freely)
// ─────────────────────────────────────────────────────────────────────────────
struct SimParams {
    // Domain
    float domain_w = 10.0f;   // world units wide
    float domain_h = 6.0f;    // world units tall

    // Particle seeding
    int   ppc       = 4;      // particles-per-cell per axis (ppc x ppc grid within each cell)
    float layer_pct = 0.20f;  // each of the 4 layers occupies this fraction of domain height

    // Grid (Phase 2+)
    int   grid_nx   = 80;     // grid resolution x
    int   grid_ny   = 48;     // grid resolution y

    // Simulation (Phase 3+)
    float dt        = 1e-4f;
    float gravity   = -9.8f;
};
