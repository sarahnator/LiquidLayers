#include "imgui.h" // Polyscope bundles Dear ImGui for UI panels
#include "polyscope/polyscope.h"
#include "simulation.h"
#include <iostream>

// ─────────────────────────────────────────────────────────────────────────────
//  Global state (simple for now — grow this as the project evolves)
// ─────────────────────────────────────────────────────────────────────────────
static Simulation *g_sim = nullptr;
static bool g_running = false; // play/pause toggle

// ─────────────────────────────────────────────────────────────────────────────
//  UI callback  — runs every frame inside Polyscope's render loop
//  This is where you'll add sliders and buttons as the project grows.
// ─────────────────────────────────────────────────────────────────────────────
void uiCallback() {
  // ── Control panel ─────────────────────────────────────────────────────────
  ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(280, 200), ImGuiCond_FirstUseEver);
  ImGui::Begin("MPM Fluid Simulation");

  ImGui::Text("Particles: %zu", g_sim->particles().size());
  ImGui::Text("Frame:     %d", g_sim->frameCount());
  ImGui::Separator();

  // Play / Pause
  if (ImGui::Button(g_running ? "Pause" : "Play ")) {
    g_running = !g_running;
  }
  ImGui::SameLine();
  // Reset
  if (ImGui::Button("Reset")) {
    g_running = false;
    g_sim->initialize();
    g_sim->updatePolyscope();
  }

  ImGui::Separator();
  ImGui::TextWrapped("Phase 1: Visualization scaffold.\n"
                     "Implement simulation.step() in\n"
                     "Phase 3 to see the fluid move.");

  ImGui::End();

  // ── Simulation tick ───────────────────────────────────────────────────────
  if (g_running) {
    // Phase 3+: uncomment this line and implement Simulation::step()
    // g_sim->step();

    // For now, a placeholder: add a tiny gravity drift to demonstrate
    // the update path works end-to-end.  Remove once step() is implemented.
    // (This is intentionally NOT real physics — just a wiring test.)
    g_sim->updatePolyscope();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
int main() {
  // ── Simulation setup ──────────────────────────────────────────────────────
  SimParams params;
  params.domain_w = 10.0f;
  params.domain_h = 6.0f;
  params.layer_pct = 0.20f; // 4 layers × 20% = 80% of domain height
  params.ppc = 4;           // 4×4 = 16 particles per grid cell → dense cloud

  Simulation sim(params);
  sim.initialize();
  g_sim = &sim;

  std::cout << "Initialized " << sim.particles().size() << " particles.\n";

  // ── Polyscope setup ───────────────────────────────────────────────────────
  polyscope::init();
  polyscope::options::programName = "MPM Fluid – Phase 1";
  polyscope::view::upDir = polyscope::UpDir::YUp;
  polyscope::view::style = polyscope::NavigateStyle::Planar; // 2D-friendly
  polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;

  // Register the particle cloud once
  sim.registerPolyscope();

  // Set camera to look at the XY plane
  // (Polyscope will auto-fit on first run; you can save the view afterwards)
  polyscope::view::lookAt(
      {5.0, 3.0, 20.0}, // camera position (above the Z=0 plane)
      {5.0, 3.0, 0.0},  // look-at target  (center of domain)
      {0.0, 1.0, 0.0}   // up direction
  );

  // Register our custom UI panel
  polyscope::state::userCallback = uiCallback;

  // ── Enter render loop (blocks until window is closed) ────────────────────
  polyscope::show();

  return 0;
}
