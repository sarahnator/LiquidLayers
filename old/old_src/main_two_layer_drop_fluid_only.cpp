#include "imgui.h"
#include "polyscope/curve_network.h"
#include "polyscope/polyscope.h"
#include "simulation_two_layer_drop_fluid_only.h"
#include <algorithm>
#include <array>
#include <string>

static TwoLayerDropSimulation *g_sim = nullptr;
static bool g_running = false;
static int g_spf = 5;

static const char *kMaterialNames[] = {"Water", "Soil", "Sand", "Rock"};

static void makeFluid(MaterialParams &mp, float rho, float K, float gamma,
                      float viscosity) {
  mp.model = ConstitutiveModel::WeaklyCompressibleFluid;
  mp.density0 = rho;
  mp.bulk_modulus = K;
  mp.gamma = gamma;
  mp.viscosity = viscosity;
  mp.computeDerived();
}

static void applyDefaultTwoLayerPreset(TwoLayerDropSimulation &sim) {
  auto &scene = sim.sceneParamsMutable();

  scene.block_w = 4.5f;
  scene.block_h = 3.2f;
  scene.block_center_x = 5.0f;
  scene.block_center_y = 4.1f;
  scene.layer_gap = 0.02f;
  scene.buoyancy_strength = 1.25f;
  scene.phase_separation_strength = 2.5f;
  scene.bottom_material = MaterialType::Rock;
  scene.top_material = MaterialType::Water;

  makeFluid(sim.materialParamsMutable(MaterialType::Water), 1000.f, 140.f, 4.f,
            0.010f);
  makeFluid(sim.materialParamsMutable(MaterialType::Soil), 1600.f, 180.f, 4.f,
            0.020f);
  makeFluid(sim.materialParamsMutable(MaterialType::Sand), 1800.f, 180.f, 4.f,
            0.015f);
  makeFluid(sim.materialParamsMutable(MaterialType::Rock), 2200.f, 260.f, 4.f,
            0.030f);

  sim.materialStyleMutable(MaterialType::Water).color = {0.12f, 0.50f, 0.95f};
  sim.materialStyleMutable(MaterialType::Soil).color = {0.58f, 0.36f, 0.18f};
  sim.materialStyleMutable(MaterialType::Sand).color = {0.89f, 0.77f, 0.42f};
  sim.materialStyleMutable(MaterialType::Rock).color = {0.42f, 0.42f, 0.46f};

  sim.toggles.enable_gravity = true;
  sim.toggles.enable_stress = true;
  sim.toggles.enable_viscosity = true;
  sim.toggles.enable_buoyancy_helper = false;
  sim.toggles.enable_phase_separation_helper = false;
  // sim.toggles.model_water_tait = true;
  sim.toggles.bc_left = true;
  sim.toggles.bc_right = true;
  sim.toggles.bc_bottom = true;
  sim.toggles.bc_top = false;
}

static void applyTwoLayerDensityInversionPreset(TwoLayerDropSimulation &sim) {
  auto &scene = sim.sceneParamsMutable();

  scene.block_w = 4.5f;
  scene.block_h = 3.2f;
  scene.block_center_x = 5.0f;
  scene.block_center_y = 4.1f;
  scene.layer_gap = 0.02f;
  scene.buoyancy_strength = 1.25f;
  scene.phase_separation_strength = 2.5f;
  scene.bottom_material = MaterialType::Rock;
  scene.top_material = MaterialType::Water;

  makeFluid(sim.materialParamsMutable(MaterialType::Water), 2200.f, 120.f, 4.f,
            0.006f);
  makeFluid(sim.materialParamsMutable(MaterialType::Rock), 700.f, 120.f, 4.f,
            0.006f);

  // Keep the two unused slots fluid-editable as well, so the user can switch
  // the dropdowns to them without needing another preset.
  makeFluid(sim.materialParamsMutable(MaterialType::Soil), 1300.f, 140.f, 4.f,
            0.010f);
  makeFluid(sim.materialParamsMutable(MaterialType::Sand), 1600.f, 150.f, 4.f,
            0.012f);

  sim.materialStyleMutable(MaterialType::Water).color = {0.16f, 0.42f, 0.95f};
  sim.materialStyleMutable(MaterialType::Soil).color = {0.55f, 0.33f, 0.18f};
  sim.materialStyleMutable(MaterialType::Sand).color = {0.87f, 0.76f, 0.45f};
  sim.materialStyleMutable(MaterialType::Rock).color = {0.34f, 0.34f, 0.38f};

  sim.toggles.enable_gravity = true;
  sim.toggles.enable_stress = true;
  sim.toggles.enable_viscosity = true;
  sim.toggles.enable_buoyancy_helper = false;
  sim.toggles.enable_phase_separation_helper = false;
  sim.toggles.enable_vol_recompute = false;
  // sim.toggles.model_water_tait = true;
  sim.toggles.bc_left = true;
  sim.toggles.bc_right = true;
  sim.toggles.bc_bottom = true;
  sim.toggles.bc_top = false;
}

static void refreshAfterReinitialize(TwoLayerDropSimulation &sim) {
  polyscope::removeStructure("two_layer_drop_particles", false);
  sim.registerPolyscope();
}

static void registerBoxBoundary(double xmin, double xmax, double ymin,
                                double ymax) {
  Eigen::MatrixXd nodes(4, 3);
  nodes << xmin, ymin, 0, xmax, ymin, 0, xmax, ymax, 0, xmin, ymax, 0;

  Eigen::MatrixXi edges(4, 2);
  edges << 0, 1, 1, 2, 2, 3, 3, 0;

  auto *box = polyscope::registerCurveNetwork("box boundary", nodes, edges);
  box->setRadius(0.002);
  box->setColor({0.0, 0.0, 0.0});
}

static void showParticleInfo(int idx) {
  if (idx < 0 || idx >= static_cast<int>(g_sim->particles().size()))
    return;
  const auto &p = g_sim->particles()[idx];
  const auto &mp = g_sim->materialParams(p.material);

  ImGui::Text("material : %s", kMaterialNames[(int)p.material]);
  ImGui::Text("pos      : (%.3f, %.3f)", p.pos.x(), p.pos.y());
  ImGui::Text("vel      : (%.3f, %.3f)", p.vel.x(), p.vel.y());
  ImGui::Text("J=det(F) : %.4f", p.F.determinant());
  ImGui::Text("F        : [[%.2f %.2f][%.2f %.2f]]", p.F(0, 0), p.F(0, 1),
              p.F(1, 0), p.F(1, 1));
  ImGui::Separator();
  ImGui::Text("rho0     : %.1f", mp.density0);
  ImGui::Text("model    : Newtonian fluid");
  ImGui::Text("bulk     : %.3e", mp.bulk_modulus);
  ImGui::Text("gamma    : %.3f", mp.gamma);
  ImGui::Text("visc     : %.4f", mp.viscosity);
}

static void showFluidMaterialEditor(const char *label, MaterialParams &mp,
                                    MaterialRenderStyle &style) {
  if (!ImGui::TreeNode(label))
    return;

  ImGui::PushID(label);

  // Hard-lock the scene to the Newtonian fluid family requested by the user.
  mp.model = ConstitutiveModel::WeaklyCompressibleFluid;

  ImGui::ColorEdit3("Color", style.color.data());
  ImGui::InputFloat("Density##density", &mp.density0);
  ImGui::InputFloat("Bulk modulus##K", &mp.bulk_modulus, 0.f, 0.f, "%.3e");
  ImGui::SliderFloat("Gamma##gamma", &mp.gamma, 1.0f, 12.0f);
  ImGui::SliderFloat("Viscosity##muv", &mp.viscosity, 0.0f, 2.0f);

  mp.density0 = std::max(mp.density0, 1.f);
  mp.bulk_modulus = std::max(mp.bulk_modulus, 1e-3f);
  mp.gamma = std::clamp(mp.gamma, 1.f, 12.f);
  mp.viscosity = std::max(mp.viscosity, 0.f);
  mp.computeDerived();

  ImGui::Text("Model     : Newtonian fluid (locked)");
  ImGui::TreePop();
  ImGui::PopID();
}

void uiCallback() {
  auto &t = g_sim->toggles;
  auto &p = g_sim->paramsMutable();
  auto &scene = g_sim->sceneParamsMutable();

  ImGui::SetNextWindowPos(ImVec2(320, 10), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(430, 860), ImGuiCond_FirstUseEver);
  ImGui::Begin("Two-layer density inversion MPM debug");

  ImGui::Text("Particles : %zu", g_sim->particles().size());
  ImGui::Text("Frame     : %d", g_sim->frameCount());
  ImGui::Text("dt        : %.2e", p.dt);
  ImGui::Separator();

  if (ImGui::Button(g_running ? "Pause" : "Play"))
    g_running = !g_running;
  ImGui::SameLine();
  if (ImGui::Button("Step x1")) {
    g_sim->step();
    g_sim->updatePolyscope();
  }
  ImGui::SameLine();
  if (ImGui::Button("Step x20")) {
    for (int i = 0; i < 20; ++i)
      g_sim->step();
    g_sim->updatePolyscope();
  }
  ImGui::SliderInt("Steps / frame", &g_spf, 1, 30);

  if (ImGui::Button("Reset simulation")) {
    g_running = false;
    p.computeDerived();
    g_sim->initialize();
    refreshAfterReinitialize(*g_sim);
  }
  if (ImGui::Button("Apply default fluid preset")) {
    applyDefaultTwoLayerPreset(*g_sim);
    p.computeDerived();
    g_sim->initialize();
    refreshAfterReinitialize(*g_sim);
    g_running = false;
  }
  if (ImGui::Button("Apply density inversion preset")) {
    applyTwoLayerDensityInversionPreset(*g_sim);
    p.computeDerived();
    g_sim->initialize();
    refreshAfterReinitialize(*g_sim);
    g_running = false;
  }

  ImGui::Separator();
  ImGui::TextWrapped(
      "This version keeps the two-layer drop setup, but treats every material "
      "slot as a Newtonian fluid. You can choose the lower and upper material "
      "slots, edit each slot's color and fluid parameters, and use a much "
      "wider initial drop so the simulation starts with many more particles.");

  ImGui::Separator();
  ImGui::Checkbox("Enable gravity", &t.enable_gravity);
  ImGui::Checkbox("Enable stress", &t.enable_stress);
  ImGui::Checkbox("Enable viscosity", &t.enable_viscosity);
  ImGui::Checkbox("Buoyancy helper", &t.enable_buoyancy_helper);
  ImGui::Checkbox("Phase-separation helper", &t.enable_phase_separation_helper);
  // ImGui::Checkbox("Fluid EOS / pressure", &t.model_water_tait);

  if (ImGui::CollapsingHeader("Boundary conditions",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Checkbox("Left wall", &t.bc_left);
    ImGui::SameLine(150.f);
    ImGui::Checkbox("Right wall", &t.bc_right);
    ImGui::Checkbox("Bottom wall", &t.bc_bottom);
    ImGui::SameLine(150.f);
    ImGui::Checkbox("Top wall", &t.bc_top);
  }

  if (ImGui::CollapsingHeader("Simulation parameters",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::InputFloat("dt", &p.dt, 0.f, 0.f, "%.2e");
    p.dt = std::max(p.dt, 1e-7f);
    ImGui::InputFloat("gravity", &p.gravity);
    ImGui::Checkbox("Volume recompute", &t.enable_vol_recompute);
    ImGui::InputFloat("block width", &scene.block_w);
    ImGui::InputFloat("block height", &scene.block_h);
    ImGui::InputFloat("block center x", &scene.block_center_x);
    ImGui::InputFloat("block center y", &scene.block_center_y);
    ImGui::SliderFloat("layer gap", &scene.layer_gap, 0.0f, 0.15f);
    ImGui::SliderFloat("buoyancy strength", &scene.buoyancy_strength, 0.0f,
                       4.0f);
    ImGui::SliderFloat("phase separation", &scene.phase_separation_strength,
                       0.0f, 6.0f);

    scene.block_w = std::clamp(scene.block_w, 0.1f, p.domain_w - 0.2f);
    scene.block_h = std::clamp(scene.block_h, 0.1f, p.domain_h - 0.2f);
    scene.block_center_x =
        std::clamp(scene.block_center_x, 0.1f + 0.5f * scene.block_w,
                   p.domain_w - 0.1f - 0.5f * scene.block_w);
    scene.block_center_y =
        std::clamp(scene.block_center_y, 0.1f + 0.5f * scene.block_h,
                   p.domain_h - 0.1f - 0.5f * scene.block_h);
    scene.layer_gap = std::max(scene.layer_gap, 0.0f);
    scene.buoyancy_strength = std::max(scene.buoyancy_strength, 0.0f);
    scene.phase_separation_strength =
        std::max(scene.phase_separation_strength, 0.0f);

    int bottom = static_cast<int>(scene.bottom_material);
    int top = static_cast<int>(scene.top_material);
    ImGui::Combo("Bottom material", &bottom, kMaterialNames,
                 IM_ARRAYSIZE(kMaterialNames));
    ImGui::Combo("Top material", &top, kMaterialNames,
                 IM_ARRAYSIZE(kMaterialNames));
    scene.bottom_material = static_cast<MaterialType>(bottom);
    scene.top_material = static_cast<MaterialType>(top);
  }

  if (ImGui::CollapsingHeader("Fluid material slots",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
    showFluidMaterialEditor("Water slot",
                            g_sim->materialParamsMutable(MaterialType::Water),
                            g_sim->materialStyleMutable(MaterialType::Water));
    showFluidMaterialEditor("Soil slot",
                            g_sim->materialParamsMutable(MaterialType::Soil),
                            g_sim->materialStyleMutable(MaterialType::Soil));
    showFluidMaterialEditor("Sand slot",
                            g_sim->materialParamsMutable(MaterialType::Sand),
                            g_sim->materialStyleMutable(MaterialType::Sand));
    showFluidMaterialEditor("Rock slot",
                            g_sim->materialParamsMutable(MaterialType::Rock),
                            g_sim->materialStyleMutable(MaterialType::Rock));
    ImGui::TextWrapped(
        "Density edits affect newly initialized particle masses. Reset after "
        "large density edits so the block is re-seeded consistently. This "
        "scene exposes four editable fluid slots because the current project "
        "stores particle material using the existing MaterialType enum.");
  }

  if (ImGui::CollapsingHeader("Particle inspector")) {
    static int inspect_idx = 0;
    ImGui::InputInt("Index", &inspect_idx);
    inspect_idx = std::clamp(inspect_idx, 0,
                             std::max(0, (int)g_sim->particles().size() - 1));
    showParticleInfo(inspect_idx);
  }

  ImGui::End();

  if (g_running) {
    for (int i = 0; i < g_spf; ++i)
      g_sim->step();
    g_sim->updatePolyscope();
  }
}

int main() {
  SimParams params;
  params.domain_w = 10.f;
  params.domain_h = 6.f;
  params.ppc = 4;
  params.grid_nx = 80;
  params.grid_ny = 48;
  params.dt = 1e-5f;
  params.gravity = -9.8f;
  params.computeDerived();

  TwoLayerDropSimulation sim(params);
  applyTwoLayerDensityInversionPreset(sim);
  sim.initialize();
  g_sim = &sim;

  polyscope::init();
  registerBoxBoundary(0.f, params.domain_w, 0.f, params.domain_h);

  polyscope::options::programName =
      "Two-layer density inversion MPM debug (fluid-only slots)";
  polyscope::view::upDir = polyscope::UpDir::YUp;
  polyscope::view::style = polyscope::NavigateStyle::Planar;
  polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;

  sim.registerPolyscope();
  polyscope::view::lookAt({5., 3., 20.}, {5., 3., 0.}, {0., 1., 0.});
  polyscope::state::userCallback = uiCallback;
  polyscope::show();
  return 0;
}
