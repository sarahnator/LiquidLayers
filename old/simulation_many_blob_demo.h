#pragma once

#include "types.h"
#include <array>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
//  SimTogglesManyBlobDemo
//
//  Runtime feature flags for a stylized many-blob fluid demo intended to get
//  closer to the look of a layered / metaball liquid toy. The physics is still
//  MPM-based, but the initialization and helper forces are chosen to encourage
//  persistent large-scale fluid motion rather than a single thin puddle.
// ─────────────────────────────────────────────────────────────────────────────
struct SimTogglesManyBlobDemo {
  bool render_particles = true;

  bool enable_gravity = true;
  bool enable_stress = true;
  bool enable_viscosity = true;

  bool model_water_tait = true;
  bool model_water_freset = false;

  bool bc_left = true;
  bool bc_right = true;
  bool bc_bottom = true;
  bool bc_top = false;

  // Visual / stylized helper forces.
  bool enable_buoyancy_helper = true;
  bool enable_phase_separation_helper = true;
  bool enable_surface_tension_helper = true;
};

// ─────────────────────────────────────────────────────────────────────────────
//  ManyBlobSceneParams
//
//  Parameters controlling the number, size, and layout of the spawned fluid
//  blobs. Only two materials are used in this demo, but each keeps its own
//  MaterialType and MaterialParams so the particles can still be color-coded.
// ─────────────────────────────────────────────────────────────────────────────
struct ManyBlobSceneParams {
  int num_blobs_x = 5;
  int num_blobs_y = 4;

  float blob_radius = 0.42f;
  float blob_spacing_x = 0.95f;
  float blob_spacing_y = 0.82f;

  float blob_center_x = 3.4f;
  float blob_center_y = 6.6f;

  float buoyancy_strength = 1.25f;
  float phase_separation_strength = 2.4f;
  float surface_tension_strength = 0.85f;

  // Add slight initial swirl and downward speed so the blobs collide, merge,
  // and exchange momentum rather than simply dropping as a perfectly still
  // lattice.
  float initial_swirl_speed = 0.5f;
  float initial_downward_speed = -0.2f;
};

class ManyBlobSimulation {
public:
  explicit ManyBlobSimulation(SimParams params = {});

  void initialize();
  void step();

  void clearGrid();
  void substep_P2G();
  void substep_gridUpdate();
  void substep_G2P();
  void substep_advect();

  int gridIdx(int i, int j) const { return j * params_.grid_nx + i; }
  bool inGrid(int i, int j) const {
    return i >= 0 && i < params_.grid_nx && j >= 0 && j < params_.grid_ny;
  }

  void registerPolyscope();
  void updatePolyscope();

  const std::vector<Particle> &particles() const { return particles_; }
  const SimParams &params() const { return params_; }
  SimParams &paramsMutable() { return params_; }

  const ManyBlobSceneParams &sceneParams() const { return scene_params_; }
  ManyBlobSceneParams &sceneParamsMutable() { return scene_params_; }

  const MaterialParams &materialParams(MaterialType m) const;
  MaterialParams &materialParamsMutable(MaterialType m);

  int frameCount() const { return frame_; }

  SimTogglesManyBlobDemo toggles;

private:
  SimParams params_;
  ManyBlobSceneParams scene_params_;
  std::vector<Particle> particles_;
  std::vector<GridNode> grid_;
  int frame_ = 0;

  // We keep only two actual phases in this demo: Water and Rock. They are both
  // configured as fluids by the preset, but retain different colors and rest
  // densities. The names are just convenient labels inherited from types.h.
  MaterialParams water_params_ = defaultMaterialParams(MaterialType::Water);
  MaterialParams rock_params_ = defaultMaterialParams(MaterialType::Rock);

  // Grid-averaged phase and density fields used by the helper forces.
  std::vector<float> grid_phase_num_;
  std::vector<float> grid_rho_num_;
  std::vector<float> grid_weight_sum_;
  std::vector<float> grid_phase_avg_;
  std::vector<float> grid_rho_avg_;
  std::vector<float> grid_phase_lap_;

  static constexpr const char *kCloudName = "many_blob_particles";

  void buildRenderArrays(std::vector<std::array<double, 3>> &,
                         std::vector<std::array<double, 3>> &) const;
  void addBlob(const Eigen::Vector2f &center, float radius, MaterialType mat,
               int blob_index);

  void finalizeHelperGridScalars();
  float sampleLocalRestDensity(const Particle &p) const;
  Eigen::Vector2f samplePhaseGradient(const Particle &p) const;
  float samplePhaseLaplacian(const Particle &p) const;
  void scatterHelperBodyForce(const Particle &p,
                              const Eigen::Vector2f &accel);

  Eigen::Matrix2f kirchhoffStress(const Particle &p) const;
  Eigen::Matrix2f stressFluid(const Particle &p, const MaterialParams &mp) const;
};
