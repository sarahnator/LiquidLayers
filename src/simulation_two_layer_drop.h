#pragma once

#include "types.h"
#include <array>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
//  SimTogglesTwoLayerDrop
//
//  Runtime feature flags for the two-layer density-inversion debugging scene.
//  This scene is meant to make density-driven fluid rearrangement easier to
//  observe than in the four-layer drop.
// ─────────────────────────────────────────────────────────────────────────────
struct SimTogglesTwoLayerDrop {
  bool render_particles = true;

  bool enable_gravity = true;
  bool enable_stress = true;
  bool enable_viscosity = true;

  // Extra visual/behavior helpers for the two-layer test. These are not meant
  // to be physically exact. Instead, they bias the motion toward the kind of
  // buoyant inversion / phase separation that is easy to see in an interactive
  // liquid-layers demo.
  bool enable_buoyancy_helper = true;
  bool enable_phase_separation_helper = true;

  // Keep the same fluid toggles as the other debugging scenes so the user can
  // easily compare behaviors across executables.
  bool model_water_tait = true;

  bool bc_left = true;
  bool bc_right = true;
  bool bc_bottom = true;
  bool bc_top = false;
};

// ─────────────────────────────────────────────────────────────────────────────
//  TwoLayerDropSceneParams
//
//  Controls the initial geometry for the dropped two-layer block.
// ─────────────────────────────────────────────────────────────────────────────
struct TwoLayerDropSceneParams {
  float block_w = 0.75f;
  float block_h = 3.2f;
  float block_center_x = 5.0f;
  float block_center_y = 4.1f;

  // Optional empty spacing inserted between the two bands inside the dropped
  // block. A small nonzero gap makes inversion and splash easier to observe.
  float layer_gap = 0.02f;

  // Helper-force strengths. These are dimensionless tuning parameters used by
  // the two-layer debugging scene only.
  float buoyancy_strength = 1.25f;
  float phase_separation_strength = 2.5f;

  // The lower and upper materials used by the two-layer initializer.
  MaterialType bottom_material = MaterialType::Rock;
  MaterialType top_material = MaterialType::Water;
};

class TwoLayerDropSimulation {
public:
  explicit TwoLayerDropSimulation(SimParams params = {});

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

  const TwoLayerDropSceneParams &sceneParams() const { return scene_params_; }
  TwoLayerDropSceneParams &sceneParamsMutable() { return scene_params_; }

  const MaterialParams &materialParams(MaterialType m) const;
  MaterialParams &materialParamsMutable(MaterialType m);

  int frameCount() const { return frame_; }

  SimTogglesTwoLayerDrop toggles;

private:
  SimParams params_;
  TwoLayerDropSceneParams scene_params_;
  std::vector<Particle> particles_;
  std::vector<GridNode> grid_;
  int frame_ = 0;

  // Smoothed scalar fields used by the optional helper forces.
  std::vector<float> grid_phase_num_;
  std::vector<float> grid_rho_num_;
  std::vector<float> grid_weight_sum_;
  std::vector<float> grid_phase_avg_;
  std::vector<float> grid_rho_avg_;

  MaterialParams water_params_ = defaultMaterialParams(MaterialType::Water);
  MaterialParams soil_params_ = defaultMaterialParams(MaterialType::Soil);
  MaterialParams sand_params_ = defaultMaterialParams(MaterialType::Sand);
  MaterialParams rock_params_ = defaultMaterialParams(MaterialType::Rock);

  static constexpr const char *kCloudName = "two_layer_drop_particles";

  void buildRenderArrays(std::vector<std::array<double, 3>> &,
                         std::vector<std::array<double, 3>> &) const;
  void addTwoLayerBlock(float x_min, float x_max, float y_min, float y_max,
                        MaterialType bottom_mat, MaterialType top_mat);
  float phaseSign(MaterialType m) const;
  void finalizeHelperGridScalars();
  float sampleLocalRestDensity(const Particle &p) const;
  Eigen::Vector2f samplePhaseGradient(const Particle &p) const;
  void scatterHelperBodyForce(const Particle &p, const Eigen::Vector2f &accel);

  Eigen::Matrix2f kirchhoffStress(const Particle &p) const;
  Eigen::Matrix2f stressFluid(const Particle &p,
                              const MaterialParams &mp) const;
  Eigen::Matrix2f stressFixedCorotated(const Particle &p,
                                       const MaterialParams &mp) const;
  Eigen::Matrix2f stressDruckerPrager(const Particle &p,
                                      const MaterialParams &mp) const;
  void projectDruckerPrager(Particle &p, const MaterialParams &mp) const;
};
