#include "imgui.h"
#include "polyscope/curve_network.h"
#include "polyscope/polyscope.h"
#include "simulation_incompressible_blob_tank.h"

#include <algorithm>
#include <string>

static IncompressibleBlobTankSimulation* g_sim = nullptr;
static bool g_running = false;
static int g_spf = 5;

static void registerTankBoundary(double xmin, double xmax, double ymin,
                                 double ymax) {
  Eigen::MatrixXd nodes(4, 3);
  nodes << xmin, ymin, 0, xmax, ymin, 0, xmax, ymax, 0, xmin, ymax, 0;
  Eigen::MatrixXi edges(4, 2);
  edges << 0, 1, 1, 2, 2, 3, 3, 0;

  auto* box = polyscope::registerCurveNetwork("tank boundary", nodes, edges);
  box->setRadius(0.0018);
  box->setColor({0.0, 0.0, 0.0});
}

static void applyGrantLikePreset(IncompressibleBlobTankSimulation& sim) {
  auto& p = sim.paramsMutable();
  auto& s = sim.sceneParamsMutable();
  auto& t = sim.toggles;

  p.domain_w = 6.f;
  p.domain_h = 10.f;
  p.grid_nx = 84;
  p.grid_ny = 140;
  p.ppc = 4;
  p.dt = 4e-5f;
  p.gravity = -4.5f;
  p.computeDerived();

  s.tank_xmin = 1.8f;
  s.tank_xmax = 4.2f;
  s.tank_ymin = 0.f;
  s.tank_ymax = 10.f;
  s.spawn_ymax = 2.8f;
  s.blob_rows = 9;
  s.blob_cols = 4;
  s.blob_radius = 0.23f;
  s.blob_jitter = 0.24f;
  s.void_probability = 0.15f;
  s.phase_a_density = 850.f;
  s.phase_b_density = 1500.f;
  s.phase_a_color = {0.19f, 0.58f, 0.97f};
  s.phase_b_color = {0.95f, 0.24f, 0.45f};
  s.pressure_iters = 70;
  s.surface_tension = 2.4f;
  s.cohesion = 1.7f;
  s.stirring_strength = 15.f;
  s.stirring_radius = 0.85f;
  s.stirring_frequency = 0.40f;
  s.stirring_y = 4.3f;

  t.enable_gravity = true;
  t.enable_projection = true;
  t.enable_surface_tension = true;
  t.enable_cohesion = true;
  t.enable_stirring = false;
  t.bc_left = true;
  t.bc_right = true;
  t.bc_bottom = true;
  t.bc_top = false;
  t.no_slip_walls = true;
}

static void rebuildSim() {
  g_running = false;
  g_sim->paramsMutable().computeDerived();
  g_sim->initialize();
  g_sim->updatePolyscope(true);
}

void uiCallback() {
  auto& p = g_sim->paramsMutable();
  auto& s = g_sim->sceneParamsMutable();
  auto& t = g_sim->toggles;

  ImGui::SetNextWindowPos(ImVec2(320, 10), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(420, 860), ImGuiCond_FirstUseEver);
  ImGui::Begin("Incompressible blob tank");

  ImGui::Text("Particles : %zu", g_sim->particles().size());
  ImGui::Text("Frame     : %d", g_sim->frameCount());
  ImGui::Text("time      : %.3f", g_sim->simTime());
  ImGui::Separator();

  if (ImGui::Button(g_running ? "Pause" : "Play"))
    g_running = !g_running;
  ImGui::SameLine();
  if (ImGui::Button("Step x1")) {
    g_sim->step();
    g_sim->updatePolyscope();
  }
  ImGui::SameLine();
  if (ImGui::Button("Step x50")) {
    for (int i = 0; i < 50; ++i)
      g_sim->step();
    g_sim->updatePolyscope();
  }
  ImGui::SliderInt("Steps / frame", &g_spf, 1, 40);

  if (ImGui::Button("Reset simulation"))
    rebuildSim();
  ImGui::SameLine();
  if (ImGui::Button("Apply Grant-like preset")) {
    applyGrantLikePreset(*g_sim);
    rebuildSim();
  }

  ImGui::Separator();
  ImGui::TextWrapped(
      "This demo uses an approximate incompressible MPM liquid variant: the "
      "particles transfer momentum to a background grid, external/body forces "
      "are applied on the grid, then a grid pressure projection reduces "
      "divergence before G2P.  The extra surface tension / cohesion / stirring "
      "toggles are appearance-first helper forces for a Grant-Kot-inspired "
      "liquid-layers look.");

  if (ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::InputFloat("dt", &p.dt, 0.f, 0.f, "%.2e");
    p.dt = std::max(p.dt, 1e-6f);
    ImGui::InputFloat("gravity", &p.gravity);
    ImGui::InputInt("pressure iterations", &s.pressure_iters);
    s.pressure_iters = std::max(s.pressure_iters, 1);
  }

  if (ImGui::CollapsingHeader("Tank / initialization",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::InputFloat("tank xmin", &s.tank_xmin);
    ImGui::InputFloat("tank xmax", &s.tank_xmax);
    ImGui::InputFloat("spawn ymax", &s.spawn_ymax);
    ImGui::InputInt("blob rows", &s.blob_rows);
    ImGui::InputInt("blob cols", &s.blob_cols);
    ImGui::InputFloat("blob radius", &s.blob_radius);
    ImGui::SliderFloat("blob jitter", &s.blob_jitter, 0.f, 0.5f);
    ImGui::SliderFloat("void probability", &s.void_probability, 0.f, 0.6f);
    s.blob_rows = std::max(s.blob_rows, 1);
    s.blob_cols = std::max(s.blob_cols, 1);
    s.blob_radius = std::max(s.blob_radius, 0.05f);
    s.tank_xmax = std::max(s.tank_xmax, s.tank_xmin + 0.3f);
    s.spawn_ymax = std::clamp(s.spawn_ymax, s.tank_ymin + 0.3f, s.tank_ymax - 0.3f);
  }

  if (ImGui::CollapsingHeader("Two fluids", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::InputFloat("Phase A density", &s.phase_a_density);
    ImGui::InputFloat("Phase B density", &s.phase_b_density);
    ImGui::ColorEdit3("Phase A color", s.phase_a_color.data());
    ImGui::ColorEdit3("Phase B color", s.phase_b_color.data());
    s.phase_a_density = std::max(s.phase_a_density, 1.f);
    s.phase_b_density = std::max(s.phase_b_density, 1.f);
    ImGui::TextWrapped(
        "Phase A uses the Water particle label internally and Phase B uses "
        "the Rock label, but both are treated as liquids in this scene.");
  }

  if (ImGui::CollapsingHeader("Forces / helpers",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Checkbox("Enable gravity", &t.enable_gravity);
    ImGui::Checkbox("Incompressible projection", &t.enable_projection);
    ImGui::Checkbox("Surface tension", &t.enable_surface_tension);
    ImGui::Checkbox("Cohesion", &t.enable_cohesion);
    ImGui::Checkbox("Periodic stirring", &t.enable_stirring);
    ImGui::SliderFloat("surface tension strength", &s.surface_tension, 0.f, 8.f);
    ImGui::SliderFloat("cohesion strength", &s.cohesion, 0.f, 6.f);
    ImGui::SliderFloat("stirring strength", &s.stirring_strength, 0.f, 40.f);
    ImGui::SliderFloat("stirring radius", &s.stirring_radius, 0.1f, 2.0f);
    ImGui::SliderFloat("stirring frequency", &s.stirring_frequency, 0.05f, 2.0f);
    ImGui::SliderFloat("stirring y", &s.stirring_y, s.tank_ymin + 0.5f,
                       s.tank_ymax - 0.5f);
  }

  if (ImGui::CollapsingHeader("Boundary conditions",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Checkbox("Left wall", &t.bc_left);
    ImGui::SameLine(140.f);
    ImGui::Checkbox("Right wall", &t.bc_right);
    ImGui::Checkbox("Bottom wall", &t.bc_bottom);
    ImGui::SameLine(140.f);
    ImGui::Checkbox("Top wall", &t.bc_top);
    ImGui::Checkbox("No-slip-ish walls", &t.no_slip_walls);
  }

  if (ImGui::CollapsingHeader("What to expect")) {
    ImGui::TextWrapped(
        "Projection ON should keep the liquid much less compressible than the "
        "earlier Tait-EOS demos. Surface tension rounds and smooths the "
        "interface, cohesion helps blobs stay thicker, and periodic stirring "
        "keeps the layers moving instead of settling into a static puddle. "
        "Reset after changing densities or initialization parameters.");
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
  params.domain_w = 6.f;
  params.domain_h = 10.f;
  params.ppc = 4;
  params.grid_nx = 84;
  params.grid_ny = 140;
  params.dt = 4e-5f;
  params.gravity = -4.5f;
  params.computeDerived();

  IncompressibleBlobTankSimulation sim(params);
  applyGrantLikePreset(sim);
  sim.initialize();
  g_sim = &sim;

  polyscope::init();
  const auto& s = sim.sceneParams();
  registerTankBoundary(s.tank_xmin, s.tank_xmax, s.tank_ymin, s.tank_ymax);

  polyscope::options::programName = "Incompressible blob tank";
  polyscope::view::upDir = polyscope::UpDir::YUp;
  polyscope::view::style = polyscope::NavigateStyle::Planar;
  polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;

  sim.registerPolyscope();
  polyscope::view::lookAt({3.0, 5.0, 20.0}, {3.0, 5.0, 0.0}, {0.0, 1.0, 0.0});
  polyscope::state::userCallback = uiCallback;
  polyscope::show();
  return 0;
}
