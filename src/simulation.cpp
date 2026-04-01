#include "simulation.h"
#include "polyscope/polyscope.h"
#include "polyscope/point_cloud.h"
#include <cmath>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
Simulation::Simulation(SimParams params) : params_(std::move(params)) {}

// ─────────────────────────────────────────────────────────────────────────────
//  initialize()
//
//  Seeds particles on a regular grid within each horizontal layer.
//  The domain spans [0, domain_w] x [0, domain_h].
//  Layers are stacked from bottom (rock) to top (water).
//
//  Jitter is added so the initial configuration doesn't look like a crystal.
// ─────────────────────────────────────────────────────────────────────────────
void Simulation::initialize() {
    particles_.clear();

    const float W    = params_.domain_w;
    const float H    = params_.domain_h;
    const float pct  = params_.layer_pct;   // fraction of H per layer
    const int   ppc  = params_.ppc;

    // Layer layout (bottom → top in Y)
    // Each layer occupies [y_lo, y_hi)
    struct Layer { MaterialType mat; float y_lo; float y_hi; };
    std::vector<Layer> layers = {
        { MaterialType::Rock,  0.0f * H * pct, 1.0f * H * pct },
        { MaterialType::Sand,  1.0f * H * pct, 2.0f * H * pct },
        { MaterialType::Soil,  2.0f * H * pct, 3.0f * H * pct },
        { MaterialType::Water, 3.0f * H * pct, 4.0f * H * pct },
    };

    // Cell size used for spacing — roughly matches grid resolution
    const float dx = W / static_cast<float>(params_.grid_nx);
    const float dy = (H * 4.0f * pct) / static_cast<float>(params_.grid_ny);
    const float px = dx / static_cast<float>(ppc);  // spacing between particles
    const float py = dy / static_cast<float>(ppc);

    // Simple deterministic jitter (Halton-like offset per particle)
    auto jitter = [&](int i, float scale) -> float {
        // Very small hash-based offset — keeps it reproducible
        return scale * (static_cast<float>((i * 1013904223 + 1664525) & 0xFFFF) / 65535.0f - 0.5f);
    };

    int pidx = 0;
    for (const auto& layer : layers) {
        int iy = 0;
        for (float y = layer.y_lo + py * 0.5f; y < layer.y_hi; y += py, ++iy) {
            int ix = 0;
            for (float x = px * 0.5f; x < W; x += px, ++ix) {
                Particle p;
                p.pos.x()  = x + jitter(pidx * 2,     px * 0.3f);
                p.pos.y()  = y + jitter(pidx * 2 + 1, py * 0.3f);
                // Clamp inside domain
                p.pos.x()  = std::clamp(p.pos.x(), 0.001f, W - 0.001f);
                p.pos.y()  = std::clamp(p.pos.y(), 0.001f, H - 0.001f);
                p.material = layer.mat;
                particles_.push_back(p);
                ++pidx;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  buildRenderArrays()  — convert particle data to Polyscope-friendly arrays
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
        // Polyscope is 3D — place particles in the XY plane at Z=0
        pos3d[i]  = { static_cast<double>(p.pos.x()),
                      static_cast<double>(p.pos.y()),
                      0.0 };
        auto c    = materialColor(p.material);
        colors[i] = { c[0], c[1], c[2] };
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  registerPolyscope()  — called once after initialize()
// ─────────────────────────────────────────────────────────────────────────────
void Simulation::registerPolyscope() {
    std::vector<std::array<double,3>> pos3d, colors;
    buildRenderArrays(pos3d, colors);

    auto* cloud = polyscope::registerPointCloud(kCloudName, pos3d);

    // Visual settings — feel free to tweak
    cloud->setPointRadius(0.003);      // relative to scene bounding box
    cloud->setPointRenderMode(polyscope::PointRenderMode::Sphere);

    // Register per-particle color as a named quantity
    cloud->addColorQuantity("material_color", colors)->setEnabled(true);
}

// ─────────────────────────────────────────────────────────────────────────────
//  updatePolyscope()  — called every frame to sync positions + colors
// ─────────────────────────────────────────────────────────────────────────────
void Simulation::updatePolyscope() {
    std::vector<std::array<double,3>> pos3d, colors;
    buildRenderArrays(pos3d, colors);

    auto* cloud = polyscope::getPointCloud(kCloudName);
    cloud->updatePointPositions(pos3d);
    cloud->addColorQuantity("material_color", colors)->setEnabled(true);

    ++frame_;
}
