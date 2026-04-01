#pragma once
#include "types.h"
#include <vector>
#include <string>

// ─────────────────────────────────────────────────────────────────────────────
//  Simulation class
//
//  Owns all particle and grid data.
//  Phase 1:  initialize() + polyscope registration only.
//  Phase 2+: add step() which runs the P2G → grid update → G2P loop.
// ─────────────────────────────────────────────────────────────────────────────
class Simulation {
public:
    explicit Simulation(SimParams params = {});

    // ── Setup ─────────────────────────────────────────────────────────────────
    // Seed particles in horizontal layers (water / soil / sand / rock from top)
    void initialize();

    // ── Simulation step (Phase 2+) ────────────────────────────────────────────
    // void step();

    // ── Polyscope interface ───────────────────────────────────────────────────
    // Register/update the particle cloud in Polyscope.
    // Call registerPolyscope() once after initialize().
    // Call updatePolyscope() each frame inside the render loop.
    void registerPolyscope();
    void updatePolyscope();

    // ── Accessors ─────────────────────────────────────────────────────────────
    const std::vector<Particle>& particles() const { return particles_; }
    const SimParams&             params()    const { return params_;    }
    int                          frameCount()const { return frame_;     }

private:
    SimParams             params_;
    std::vector<Particle> particles_;
    int                   frame_ = 0;

    // Polyscope point cloud name (stable string used for updates)
    static constexpr const char* kCloudName = "particles";

    // Helper: build Nx3 position matrix and Nx3 color matrix for Polyscope
    void buildRenderArrays(
        std::vector<std::array<double,3>>& pos3d,
        std::vector<std::array<double,3>>& colors) const;
};
