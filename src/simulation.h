#pragma once
#include "types.h"
#include <vector>
#include <array>

// ─────────────────────────────────────────────────────────────────────────────
//  SimToggles
//
//  Every runtime feature flag lives here.  main.cpp owns one instance and
//  passes a pointer to Simulation so the UI can flip toggles mid-run without
//  recompiling.  Defaults correspond to the full Phase 4 behaviour.
// ─────────────────────────────────────────────────────────────────────────────
struct SimToggles {
    // ── Phase 1 ───────────────────────────────────────────────────────────────
    bool render_particles    = true;   // draw the point cloud at all

    // ── Phase 2 ───────────────────────────────────────────────────────────────
    // (P2G / G2P / advect are always on — turning them off would stop the sim)

    // ── Phase 3 ───────────────────────────────────────────────────────────────
    bool enable_gravity      = true;   // add  m*g  to grid node forces
    bool enable_stress       = true;   // scatter  -vol*tau*grad_w  to grid
    bool enable_viscosity    = true;   // include viscous term in fluid stress

    // ── Phase 4 per-material constitutive model switches ─────────────────────
    // When a model is DISABLED, the material falls back to a pressureless gas
    // (zero stress).  This lets you isolate one material at a time.
    bool model_water_tait    = true;   // Tait EOS + F-reset for water
    bool model_water_freset  = true;   // F-reset specifically (keep to avoid blow-up)
    bool model_soil_elastic  = true;   // fixed-corotated for soil
    bool model_sand_plastic  = true;   // Drucker-Prager projection for sand
    bool model_rock_elastic  = true;   // fixed-corotated for rock

    // ── Boundary conditions ───────────────────────────────────────────────────
    bool bc_left             = true;
    bool bc_right            = true;
    bool bc_bottom           = true;
    bool bc_top              = true;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Simulation
// ─────────────────────────────────────────────────────────────────────────────
class Simulation {
public:
    explicit Simulation(SimParams params = {});

    void initialize();
    void step();

    void clearGrid();
    void substep_P2G();
    void substep_gridUpdate();
    void substep_G2P();
    void substep_advect();

    int  gridIdx(int i, int j) const { return j*params_.grid_nx + i; }
    bool inGrid (int i, int j) const {
        return i >= 0 && i < params_.grid_nx &&
               j >= 0 && j < params_.grid_ny;
    }

    void registerPolyscope();
    void updatePolyscope();

    const std::vector<Particle>& particles() const { return particles_; }
    const SimParams&             params()    const { return params_;    }
    SimParams&                   paramsMutable()   { return params_;    }
    int                          frameCount()const { return frame_;     }

    // Helper: fetch the currently-active runtime-editable parameter block for
    // a given material. This keeps the constitutive-law code compact and makes
    // it explicit that the UI, not defaultMaterialParams(), is the source of
    // truth once the simulation has started.
    const MaterialParams& materialParams(MaterialType m) const;
    MaterialParams&       materialParamsMutable(MaterialType m);

    // The UI writes directly to this struct every frame
    SimToggles toggles;

private:
    SimParams             params_;
    std::vector<Particle> particles_;
    std::vector<GridNode> grid_;
    int                   frame_ = 0;

    static constexpr const char* kCloudName = "particles";
    void buildRenderArrays(std::vector<std::array<double,3>>&,
                           std::vector<std::array<double,3>>&) const;

    Eigen::Matrix2f kirchhoffStress     (const Particle& p) const;
    Eigen::Matrix2f stressFluid         (const Particle& p,
                                         const MaterialParams& mp) const;
    Eigen::Matrix2f stressFixedCorotated(const Particle& p,
                                         const MaterialParams& mp) const;
    Eigen::Matrix2f stressDruckerPrager (const Particle& p,
                                         const MaterialParams& mp) const;
    void projectDruckerPrager(Particle& p, const MaterialParams& mp) const;
};
