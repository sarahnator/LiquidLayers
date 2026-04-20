#pragma once

#include "types.h"
#include <array>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
//  IncompressibleBlobTankToggles
//
//  Approximate incompressible 2D liquid demo built on the same particle/grid
//  APIC/MPM data structures as the earlier debugging scenes, but with a grid
//  pressure projection instead of a Tait EOS pressure term.  The intent is to
//  get closer to a Grant-Kot-style "liquid layers" toy: two colored fluids in
//  a tall narrow tank, random blob initialization near the bottom, and several
//  appearance-driven helper forces (surface tension, cohesion, stirring).
//
//  Important note:
//    This is a practical / visual incompressible MPM variant, not a research-
//    grade multiphase incompressible free-surface solver.  The pressure solve
//    is a simple Jacobi projection on the background grid, and the interface
//    forces are lightweight helper forces rather than a full sharp-interface
//    model.
// ─────────────────────────────────────────────────────────────────────────────
struct IncompressibleBlobTankToggles {
  bool render_particles = true;
  bool enable_gravity = true;
  bool enable_projection = true;
  bool enable_surface_tension = true;
  bool enable_cohesion = true;
  bool enable_stirring = false;

  bool bc_left = true;
  bool bc_right = true;
  bool bc_bottom = true;
  bool bc_top = false;
  bool no_slip_walls = true;
};

// ─────────────────────────────────────────────────────────────────────────────
//  IncompressibleBlobTankSceneParams
// ─────────────────────────────────────────────────────────────────────────────
struct IncompressibleBlobTankSceneParams {
  float tank_xmin = 1.8f;
  float tank_xmax = 4.2f;
  float tank_ymin = 0.0f;
  float tank_ymax = 10.0f;

  float spawn_ymax = 2.6f;     // blobs are initialized below this height
  int   blob_rows = 8;
  int   blob_cols = 4;
  float blob_radius = 0.22f;
  float blob_jitter = 0.18f;
  float void_probability = 0.18f; // leave some empty holes to mimic random blobs

  // Phase A uses MaterialType::Water, phase B uses MaterialType::Rock.
  float phase_a_density = 850.f;
  float phase_b_density = 1450.f;
  std::array<float, 3> phase_a_color{0.22f, 0.60f, 0.95f};
  std::array<float, 3> phase_b_color{0.95f, 0.25f, 0.45f};

  // Projection / forces.
  int   pressure_iters = 60;
  float surface_tension = 2.5f;
  float cohesion = 1.6f;
  float stirring_strength = 18.f;
  float stirring_radius = 0.9f;
  float stirring_frequency = 0.55f;
  float stirring_y = 4.0f;
};

class IncompressibleBlobTankSimulation {
public:
  explicit IncompressibleBlobTankSimulation(SimParams params = {});

  void initialize();
  void step();

  void clearGrid();
  void substep_P2G();
  void substep_addBodyForces();
  void substep_project();
  void substep_G2P();
  void substep_advect();

  int  gridIdx(int i, int j) const { return j * params_.grid_nx + i; }
  bool inGrid(int i, int j) const {
    return i >= 0 && i < params_.grid_nx && j >= 0 && j < params_.grid_ny;
  }

  void registerPolyscope();
  void updatePolyscope(bool force_rebuild = false);

  const std::vector<Particle>& particles() const { return particles_; }
  const SimParams& params() const { return params_; }
  SimParams& paramsMutable() { return params_; }

  const IncompressibleBlobTankSceneParams& sceneParams() const {
    return scene_params_;
  }
  IncompressibleBlobTankSceneParams& sceneParamsMutable() {
    return scene_params_;
  }

  int frameCount() const { return frame_; }
  float simTime() const { return time_; }

  IncompressibleBlobTankToggles toggles;

private:
  SimParams params_;
  IncompressibleBlobTankSceneParams scene_params_;
  std::vector<Particle> particles_;
  std::vector<GridNode> grid_;
  std::vector<float> pressure_;
  std::vector<float> pressure_tmp_;
  std::vector<float> divergence_;
  std::vector<float> phase_scalar_;
  std::vector<float> phase_laplacian_;
  std::vector<float> mass_a_;
  std::vector<float> mass_b_;
  int frame_ = 0;
  float time_ = 0.f;

  MaterialParams phase_a_params_ = defaultMaterialParams(MaterialType::Water);
  MaterialParams phase_b_params_ = defaultMaterialParams(MaterialType::Rock);

  static constexpr const char* kCloudName = "incompressible_blob_tank_particles";

  void buildRenderArrays(std::vector<std::array<double, 3>>&,
                         std::vector<std::array<double, 3>>&) const;
  void addRandomBlobPattern();
  void addBlob(const Eigen::Vector2f& center, float radius, MaterialType mat,
               Eigen::Vector2f initial_vel);
  void finalizePhaseFields();
  void enforceTankBounds(Particle& p) const;

  const MaterialParams& materialParams(MaterialType m) const;
  MaterialParams& materialParamsMutable(MaterialType m);

  Eigen::Vector2f samplePhaseGradient(const Particle& p) const;
  float samplePhaseLaplacian(const Particle& p) const;
  void scatterParticleBodyForce(const Particle& p,
                                const Eigen::Vector2f& accel);
};
