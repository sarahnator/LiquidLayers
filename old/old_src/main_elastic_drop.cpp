#include "imgui.h"
#include "polyscope/curve_network.h"
#include "polyscope/polyscope.h"
#include "simulation_elastic_drop.h"
#include <algorithm>
#include <string>

static Simulation *g_sim = nullptr;
static bool g_running = false;
static int g_spf = 5;

static void registerBoxBoundary(double xmin = -1.0, double xmax = 1.0,
                                double ymin = 0.0, double ymax = 2.0) {
  // 4 corners (z = 0 for 2D)
  Eigen::MatrixXd nodes(4, 3);
  nodes << xmin, ymin, 0, xmax, ymin, 0, xmax, ymax, 0, xmin, ymax, 0;

  // 4 edges (connect corners)
  Eigen::MatrixXi edges(4, 2);
  edges << 0, 1, 1, 2, 2, 3, 3, 0;

  // Register with Polyscope
  auto *box = polyscope::registerCurveNetwork("box boundary", nodes, edges);

  // Optional styling
  box->setRadius(0.002);          // thickness of lines
  box->setColor({0.0, 0.0, 0.0}); // black
}

static void showParticleInfo(int idx) {
  if (idx < 0 || idx >= static_cast<int>(g_sim->particles().size()))
    return;

  const auto &p = g_sim->particles()[idx];
  const auto &mp = g_sim->params().elastic_material;
  ImGui::Text("pos : (%.4f, %.4f)", p.pos.x(), p.pos.y());
  ImGui::Text("vel : (%.4f, %.4f)", p.vel.x(), p.vel.y());
  ImGui::Text("J   : %.6f", p.F.determinant());
  ImGui::Text("F   : [[%.3f %.3f][%.3f %.3f]]", p.F(0, 0), p.F(0, 1), p.F(1, 0),
              p.F(1, 1));
  ImGui::Separator();
  ImGui::Text("E   : %.3e", mp.youngs_modulus);
  ImGui::Text("nu  : %.3f", mp.poisson_ratio);
  ImGui::Text("rho : %.1f", mp.density0);
  ImGui::Text("mode: %s", mp.plane_stress ? "plane stress" : "plane strain");
}

void uiCallback() {
  SimToggles &t = g_sim->toggles;
  SimParams &p = g_sim->paramsMutable();
  MaterialParams &mp = p.elastic_material;

  ImGui::SetNextWindowPos(ImVec2(320, 10), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(360, 760), ImGuiCond_FirstUseEver);
  ImGui::Begin("Elastic drop MPM debug");

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
    g_sim->updatePolyscope();
  }

  ImGui::Separator();
  ImGui::TextWrapped(
      "This scene is meant to isolate one phenomenon: a single elastic body "
      "falling under gravity in a box. It is useful for checking that your "
      "P2G, stress force, G2P, deformation-gradient update, and wall handling "
      "all behave sensibly.");

  ImGui::Separator();
  ImGui::Checkbox("Enable gravity", &t.enable_gravity);
  ImGui::Checkbox("Enable elastic stress", &t.enable_stress);
  ImGui::Checkbox("Sticky walls", &t.sticky_walls);

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
    ImGui::InputFloat("drop width", &p.drop_w);
    ImGui::InputFloat("drop height", &p.drop_h);
    ImGui::InputFloat("drop center x", &p.drop_center_x);
    ImGui::InputFloat("drop center y", &p.drop_center_y);
    p.drop_w = std::max(p.drop_w, 0.1f);
    p.drop_h = std::max(p.drop_h, 0.1f);
  }

  if (ImGui::CollapsingHeader("Elastic material",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::InputFloat("density", &mp.density0);
    ImGui::InputFloat("Young's modulus", &mp.youngs_modulus, 0.f, 0.f, "%.3e");
    ImGui::SliderFloat("Poisson ratio", &mp.poisson_ratio, 0.0f, 0.45f);
    ImGui::Checkbox("Plane stress", &mp.plane_stress);
    mp.density0 = std::max(mp.density0, 1.f);
    mp.youngs_modulus = std::max(mp.youngs_modulus, 1.f);
    p.computeDerived();

    ImGui::Text("mu      = %.3e", mp.mu);
    ImGui::Text("lambda2D= %.3e", mp.lambda2D());
    ImGui::TextWrapped(
        "Changing the material values updates the constitutive law used in the "
        "stress force. Reset the simulation after a large parameter change so "
        "the initial particle masses match the new density.");
  }

  if (ImGui::CollapsingHeader("Particle inspector")) {
    static int inspect_idx = 0;
    ImGui::InputInt("Index", &inspect_idx);
    inspect_idx = std::clamp(
        inspect_idx, 0,
        std::max(0, static_cast<int>(g_sim->particles().size()) - 1));
    showParticleInfo(inspect_idx);
  }

  if (ImGui::CollapsingHeader("Expected behavior guide")) {
    ImGui::TextWrapped(
        "With gravity ON and stress ON, the block should fall, hit the bottom, "
        "compress, and bounce / settle depending on the chosen stiffness and "
        "time step.\n\n"
        "With gravity ON and stress OFF, the particles behave like dust: the "
        "block loses its shape because there is no elastic restoring force.\n\n"
        "With sticky walls ON, inward wall velocity is zeroed on the grid, so "
        "the block stays inside the box.\n\n"
        "If the elastic block wildly explodes or tunnels, the first things to "
        "check are dt, stress assembly, the F update, and wall handling.");
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
  params.dt = 5e-5f;
  params.gravity = -9.8f;
  params.drop_w = 1.2f;
  params.drop_h = 1.2f;
  params.drop_center_x = 5.0f;
  params.drop_center_y = 4.6f;
  params.elastic_material.density0 = 1100.f;
  params.elastic_material.youngs_modulus = 2.0e4f;
  params.elastic_material.poisson_ratio = 0.30f;
  params.elastic_material.plane_stress = false;
  params.computeDerived();

  Simulation sim(params);
  sim.initialize();
  g_sim = &sim;

  polyscope::init();
  registerBoxBoundary(0.f, params.domain_w, 0.f, params.domain_h);
  polyscope::options::programName = "Elastic drop MPM debug";
  polyscope::view::upDir = polyscope::UpDir::YUp;
  polyscope::view::style = polyscope::NavigateStyle::Planar;
  polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;

  sim.registerPolyscope();
  polyscope::view::lookAt({5., 3., 20.}, {5., 3., 0.}, {0., 1., 0.});
  polyscope::state::userCallback = uiCallback;
  polyscope::show();
  return 0;
}
