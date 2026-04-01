#pragma once
#include "types.h"
#include <vector>
#include <string>
#include <functional>

// ─────────────────────────────────────────────────────────────────────────────
//  Simulation
// ─────────────────────────────────────────────────────────────────────────────
class Simulation {
public:
    explicit Simulation(SimParams params = {});

    // ── Setup ─────────────────────────────────────────────────────────────────
    void initialize();

    // ── Simulation step  (Phase 2: P2G + G2P skeleton) ───────────────────────
    void step();

    // ── Grid helpers ──────────────────────────────────────────────────────────
    // Flat 1D index from 2D grid indices (row-major: j * nx + i)
    int  gridIdx(int i, int j) const { return j * params_.grid_nx + i; }

    // Check whether node (i,j) is inside the grid
    bool inGrid(int i, int j) const {
        return i >= 0 && i < params_.grid_nx &&
               j >= 0 && j < params_.grid_ny;
    }

    // Reset all grid nodes to zero (called at start of every step)
    void clearGrid();

    // ── The four MPM sub-steps ────────────────────────────────────────────────
    // Exposed publicly so you can call them one-at-a-time from the UI
    // while debugging — invaluable for understanding what each step does.
    void substep_P2G();          // Particles → Grid  (mass + momentum)
    void substep_gridUpdate();   // Apply forces + gravity on grid
    void substep_G2P();          // Grid → Particles  (velocity + APIC C)
    void substep_advect();       // Move particles with their new velocities

    // ── Polyscope interface ───────────────────────────────────────────────────
    void registerPolyscope();
    void updatePolyscope();

    // ── Accessors ─────────────────────────────────────────────────────────────
    const std::vector<Particle>& particles() const { return particles_; }
    const std::vector<GridNode>& grid()      const { return grid_;      }
    const SimParams&             params()    const { return params_;    }
    int                          frameCount()const { return frame_;     }

private:
    SimParams             params_;
    std::vector<Particle> particles_;
    std::vector<GridNode> grid_;      // size = grid_nx * grid_ny
    int                   frame_ = 0;

    static constexpr const char* kCloudName = "particles";

    void buildRenderArrays(
        std::vector<std::array<double,3>>& pos3d,
        std::vector<std::array<double,3>>& colors) const;
};
