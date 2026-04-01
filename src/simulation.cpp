#include "simulation.h"
#include "polyscope/polyscope.h"
#include "polyscope/point_cloud.h"
#include <cmath>
#include <algorithm>
#include <iostream>

// ═════════════════════════════════════════════════════════════════════════════
//
//  PRACTICE FILE — implement the four functions marked TODO below.
//
//  Everything else (initialization, Polyscope rendering, the step() driver)
//  is already complete.  Build and run at any time — the simulation will
//  silently do nothing until your implementations are correct, and then
//  particles will start falling under gravity.
//
//  Recommended order:
//    1. bsplineWeight / bsplineWeightGrad  (in types.h — but test here first)
//    2. substep_P2G
//    3. substep_gridUpdate
//    4. substep_G2P
//    5. substep_advect   (given — it's trivial, read it to understand the loop)
//
//  Useful identities to keep nearby:
//    grid node world position:  x_i = i * dx,   x_j = j * dx
//    flat grid index:           gridIdx(i, j) = j * nx + i
//    WeightStencil usage:       see types.h — construct one per particle,
//                               then loop a=0..2, b=0..2 for the 3x3 stencil
//    APIC D_p^{-1}:             params_.D_inv = 4 / dx^2   (precomputed)
//
// ═════════════════════════════════════════════════════════════════════════════

Simulation::Simulation(SimParams params) : params_(std::move(params)) {
    params_.computeDerived();
    grid_.resize(params_.grid_nx * params_.grid_ny);
}

// ─────────────────────────────────────────────────────────────────────────────
//  initialize()  — COMPLETE.  Seeds particles in horizontal layers.
//  Read this to understand the data layout; don't modify it.
// ─────────────────────────────────────────────────────────────────────────────
void Simulation::initialize() {
    particles_.clear();
    frame_ = 0;

    const float W   = params_.domain_w;
    const float H   = params_.domain_h;
    const float pct = params_.layer_pct;
    const int   ppc = params_.ppc;
    const float dx  = params_.dx;

    struct Layer { MaterialType mat; float y_lo; float y_hi; float density; };
    std::vector<Layer> layers = {
        { MaterialType::Rock,  0.0f*H*pct, 1.0f*H*pct, 2500.f },
        { MaterialType::Sand,  1.0f*H*pct, 2.0f*H*pct, 1600.f },
        { MaterialType::Soil,  2.0f*H*pct, 3.0f*H*pct, 1300.f },
        { MaterialType::Water, 3.0f*H*pct, 4.0f*H*pct, 1000.f },
    };

    const float px = dx / static_cast<float>(ppc);
    const float py = dx / static_cast<float>(ppc);

    auto jitter = [&](int i, float scale) -> float {
        return scale * (static_cast<float>((i*1013904223 + 1664525) & 0xFFFF) / 65535.f - 0.5f);
    };

    int pidx = 0;
    for (const auto& layer : layers) {
        for (float y = layer.y_lo + py*0.5f; y < layer.y_hi; y += py) {
            for (float x = px*0.5f; x < W; x += px) {
                Particle p;
                p.pos.x()  = x + jitter(pidx*2,   px*0.3f);
                p.pos.y()  = y + jitter(pidx*2+1, py*0.3f);
                p.pos.x()  = std::clamp(p.pos.x(), 0.001f, W-0.001f);
                p.pos.y()  = std::clamp(p.pos.y(), 0.001f, H-0.001f);
                p.material = layer.mat;
                p.vel      = Eigen::Vector2f::Zero();
                p.C        = Eigen::Matrix2f::Zero();
                p.F        = Eigen::Matrix2f::Identity();
                p.density0 = layer.density;
                p.mass     = layer.density * px * py;
                p.vol0     = px * py;
                particles_.push_back(p);
                ++pidx;
            }
        }
    }

    std::cout << "[MPM] " << particles_.size() << " particles, grid "
              << params_.grid_nx << "x" << params_.grid_ny
              << " dx=" << params_.dx << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  clearGrid()  — COMPLETE.  Zeros every grid node.
// ─────────────────────────────────────────────────────────────────────────────
void Simulation::clearGrid() {
    for (auto& node : grid_) {
        node.mass     = 0.f;
        node.momentum = Eigen::Vector2f::Zero();
        node.vel      = Eigen::Vector2f::Zero();
        node.vel_new  = Eigen::Vector2f::Zero();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  TODO 1 of 4 — substep_P2G
//
//  Goal: scatter each particle's mass and momentum onto the 3x3 grid stencil
//        surrounding it.
//
//  For each particle p:
//    Construct a WeightStencil ws(p.pos, dx, nx, ny).
//    Loop over the 3x3 stencil: a in {0,1,2},  b in {0,1,2}.
//      Compute grid node indices:  ni = ws.base_i + a
//                                  nj = ws.base_j + b
//      Skip if !inGrid(ni, nj).
//      Get weight:  float w = ws.weight(a, b)
//      Compute the vector FROM the particle TO this grid node:
//          xip = (node world pos) - p.pos
//          where node world pos = { ni * dx,  nj * dx }
//      Accumulate onto grid_[gridIdx(ni, nj)]:
//          node.mass     += w * p.mass
//          node.momentum += w * p.mass * (p.vel + p.C * xip)
//                                        ^^^^^^^^^^^^^^^^
//                                        APIC term: local velocity correction
//                                        using the affine matrix C.
//                                        Without it: plain PIC (too diffusive).
//
//  Sanity check: after P2G, sum all node.mass values.
//  It should equal sum of all particle masses (conservation).
// ─────────────────────────────────────────────────────────────────────────────
void Simulation::substep_P2G() {
    // TODO: implement particle-to-grid transfer
    // Hints:
    //   const float dx = params_.dx;
    //   const int   nx = params_.grid_nx;
    //   const int   ny = params_.grid_ny;
    //   for (const auto& p : particles_) { ... }
}

// ─────────────────────────────────────────────────────────────────────────────
//  TODO 2 of 4 — substep_gridUpdate
//
//  Goal: convert momentum to velocity, apply gravity, enforce wall boundaries.
//
//  For each grid node (i, j):
//    Skip if node.mass < 1e-10f  (empty node — no particles nearby).
//    Recover velocity:   node.vel = node.momentum / node.mass
//    Apply gravity:      node.vel_new = node.vel
//                        node.vel_new.y() += params_.gravity * params_.dt
//    (Phase 3 will add stress forces here too.)
//
//  Boundary conditions (sticky walls — zero the velocity INTO the wall):
//    Define wall thickness:  const int wall = 2  (nodes from each edge).
//    Left wall   (i < wall):      if vel_new.x() < 0,  set it to 0
//    Right wall  (i >= nx-wall):  if vel_new.x() > 0,  set it to 0
//    Bottom wall (j < wall):      if vel_new.y() < 0,  set it to 0
//    Top wall    (j >= ny-wall):  if vel_new.y() > 0,  set it to 0
//
//  Why only zero the inward component?  So particles slide along walls
//  freely but can't pass through them.  If you zeroed both components
//  that would be a fully sticky (no-slip) wall.
// ─────────────────────────────────────────────────────────────────────────────
void Simulation::substep_gridUpdate() {
    // TODO: implement grid velocity update + boundary conditions
    // Hints:
    //   const float dt = params_.dt;
    //   const float g  = params_.gravity;
    //   const int   nx = params_.grid_nx;
    //   const int   ny = params_.grid_ny;
    //   for (int j = 0; j < ny; ++j)
    //     for (int i = 0; i < nx; ++i) { ... }
}

// ─────────────────────────────────────────────────────────────────────────────
//  TODO 3 of 4 — substep_G2P
//
//  Goal: gather updated grid velocity back to particles, and compute the
//        APIC affine matrix C and deformation gradient F.
//
//  For each particle p:
//    Construct a WeightStencil ws(p.pos, dx, nx, ny).
//    Initialize accumulators:
//        Eigen::Vector2f v_new = Vector2f::Zero()
//        Eigen::Matrix2f C_new = Matrix2f::Zero()
//    Loop over the 3x3 stencil: a in {0,1,2},  b in {0,1,2}.
//      Skip if !inGrid(ni, nj).
//      Get weight w = ws.weight(a, b).
//      Get the updated grid velocity:  vi = grid_[gridIdx(ni,nj)].vel_new
//      Vector from particle to node:   xip = { ni*dx, nj*dx } - p.pos
//      Accumulate:
//          v_new += w * vi
//          C_new += w * (vi * xip.transpose())   ← outer product: 2x1 * 1x2 = 2x2
//    After the loop:
//        p.vel = v_new
//        p.C   = params_.D_inv * C_new
//    Update deformation gradient:
//        Eigen::Matrix2f F_inc = Matrix2f::Identity() + params_.dt * p.C
//        p.F = F_inc * p.F
//
//  Note on the outer product: in Eigen,  (vi * xip.transpose())  gives a 2x2
//  matrix where entry (r,c) = vi[r] * xip[c].  This is correct.
//
//  Note on D_inv: for the quadratic B-spline, D_p = (dx^2/4) * I, so
//  D_inv = 4/dx^2.  It's precomputed in params_.D_inv.
// ─────────────────────────────────────────────────────────────────────────────
void Simulation::substep_G2P() {
    // TODO: implement grid-to-particle transfer with APIC
    // Hints:
    //   const float dx    = params_.dx;
    //   const float D_inv = params_.D_inv;
    //   const float dt    = params_.dt;
    //   const int   nx    = params_.grid_nx;
    //   const int   ny    = params_.grid_ny;
    //   for (auto& p : particles_) { ... }
}

// ─────────────────────────────────────────────────────────────────────────────
//  substep_advect()  — COMPLETE.  Move particles forward by dt * vel.
//  Read this — it's the simplest substep and shows how the others are used.
// ─────────────────────────────────────────────────────────────────────────────
void Simulation::substep_advect() {
    const float dt = params_.dt;
    const float W  = params_.domain_w;
    const float H  = params_.domain_h;
    for (auto& p : particles_) {
        p.pos += dt * p.vel;
        p.pos.x() = std::clamp(p.pos.x(), 0.001f, W - 0.001f);
        p.pos.y() = std::clamp(p.pos.y(), 0.001f, H - 0.001f);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  step()  — COMPLETE.  Orchestrates the five substeps in order.
//  Once your three TODOs above are correct, particles will fall under gravity.
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
//  Polyscope rendering — COMPLETE, don't modify.
// ─────────────────────────────────────────────────────────────────────────────
void Simulation::buildRenderArrays(
    std::vector<std::array<double,3>>& pos3d,
    std::vector<std::array<double,3>>& colors) const
{
    const size_t N = particles_.size();
    pos3d.resize(N);
    colors.resize(N);
    for (size_t i = 0; i < N; ++i) {
        const auto& p = particles_[i];
        pos3d[i]  = { (double)p.pos.x(), (double)p.pos.y(), 0.0 };
        auto c    = materialColor(p.material);
        colors[i] = { c[0], c[1], c[2] };
    }
}

void Simulation::registerPolyscope() {
    std::vector<std::array<double,3>> pos3d, colors;
    buildRenderArrays(pos3d, colors);
    auto* cloud = polyscope::registerPointCloud(kCloudName, pos3d);
    cloud->setPointRadius(0.003);
    cloud->setPointRenderMode(polyscope::PointRenderMode::Sphere);
    cloud->addColorQuantity("material_color", colors)->setEnabled(true);
}

void Simulation::updatePolyscope() {
    std::vector<std::array<double,3>> pos3d, colors;
    buildRenderArrays(pos3d, colors);
    auto* cloud = polyscope::getPointCloud(kCloudName);
    cloud->updatePointPositions(pos3d);
    cloud->addColorQuantity("material_color", colors)->setEnabled(true);
}
