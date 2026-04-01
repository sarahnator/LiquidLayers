#include "simulation.h"
#include "polyscope/polyscope.h"
#include "imgui.h"
#include <iostream>
#include <string>

// ─────────────────────────────────────────────────────────────────────────────
static Simulation* g_sim     = nullptr;
static bool        g_running = false;

// Debug mode: step through each substep manually to inspect intermediate state
static bool        g_debug_mode = false;

// ─────────────────────────────────────────────────────────────────────────────
//  Helpers to display a compact vector/matrix in ImGui
// ─────────────────────────────────────────────────────────────────────────────
static void showParticleInfo(int idx) {
    if (idx < 0 || idx >= (int)g_sim->particles().size()) return;
    const auto& p = g_sim->particles()[idx];

    ImGui::Text("pos:  (%.3f, %.3f)",  p.pos.x(), p.pos.y());
    ImGui::Text("vel:  (%.3f, %.3f)",  p.vel.x(), p.vel.y());
    ImGui::Text("mass: %.4f",          p.mass);
    ImGui::Text("C:  [[%.2f, %.2f]",   p.C(0,0), p.C(0,1));
    ImGui::Text("     [%.2f, %.2f]]",  p.C(1,0), p.C(1,1));
    ImGui::Text("F det: %.4f",         p.F.determinant());
}

// ─────────────────────────────────────────────────────────────────────────────
//  UI callback
// ─────────────────────────────────────────────────────────────────────────────
void uiCallback() {
    ImGui::SetNextWindowPos( ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(300, 460), ImGuiCond_FirstUseEver);
    ImGui::Begin("MPM Fluid — Phase 2");

    // ── Stats ──────────────────────────────────────────────────────────────
    ImGui::Text("Particles : %zu",  g_sim->particles().size());
    ImGui::Text("Grid      : %d x %d", g_sim->params().grid_nx,
                                        g_sim->params().grid_ny);
    ImGui::Text("dx        : %.4f", g_sim->params().dx);
    ImGui::Text("dt        : %.5f", g_sim->params().dt);
    ImGui::Text("Frame     : %d",   g_sim->frameCount());
    ImGui::Separator();

    // ── Playback controls ─────────────────────────────────────────────────
    if (!g_debug_mode) {
        if (ImGui::Button(g_running ? " Pause " : " Play  "))
            g_running = !g_running;
        ImGui::SameLine();
        if (ImGui::Button("Step x1")) {
            g_sim->step();
            g_sim->updatePolyscope();
        }
        ImGui::SameLine();
        if (ImGui::Button("Step x10")) {
            for (int i = 0; i < 10; ++i) g_sim->step();
            g_sim->updatePolyscope();
        }
    }

    if (ImGui::Button("Reset")) {
        g_running = false;
        g_sim->initialize();
        g_sim->updatePolyscope();
    }
    ImGui::Separator();

    // ── Debug / substep mode ──────────────────────────────────────────────
    ImGui::Checkbox("Substep debug mode", &g_debug_mode);
    if (g_debug_mode) {
        g_running = false;
        ImGui::TextColored(ImVec4(1,0.8f,0.2f,1), "Manual substep control:");

        if (ImGui::Button("1. Clear grid")) {
            g_sim->clearGrid();
            ImGui::SetTooltip("Grid zeroed");
        }
        ImGui::SameLine();
        if (ImGui::Button("2. P2G")) {
            g_sim->substep_P2G();
            g_sim->updatePolyscope();
        }
        ImGui::SameLine();
        if (ImGui::Button("3. Grid update")) {
            g_sim->substep_gridUpdate();
        }
        if (ImGui::Button("4. G2P")) {
            g_sim->substep_G2P();
            g_sim->updatePolyscope();
        }
        ImGui::SameLine();
        if (ImGui::Button("5. Advect")) {
            g_sim->substep_advect();
            g_sim->updatePolyscope();
        }

        ImGui::Separator();
        ImGui::TextWrapped(
            "Run substeps in order 1→5 to complete one timestep.\n"
            "Inspect particle 0 after each to see what changed:");

        static int inspect_idx = 0;
        ImGui::InputInt("Particle index", &inspect_idx);
        showParticleInfo(inspect_idx);
    }

    ImGui::Separator();

    // ── What to expect ────────────────────────────────────────────────────
    ImGui::TextWrapped(
        "Phase 2: particles fall under gravity\n"
        "and compress at the bottom.\n"
        "No pressure yet — that's Phase 3.\n"
        "Layers will interpenetrate for now.");

    ImGui::End();

    // ── Simulation tick ───────────────────────────────────────────────────
    if (g_running && !g_debug_mode) {
        g_sim->step();
        g_sim->updatePolyscope();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
int main() {
    SimParams params;
    params.domain_w  = 10.0f;
    params.domain_h  =  6.0f;
    params.layer_pct =  0.20f;
    params.ppc       =  4;
    params.grid_nx   = 80;
    params.grid_ny   = 48;
    params.dt        = 2e-4f;
    params.gravity   = -9.8f;
    params.computeDerived();

    Simulation sim(params);
    sim.initialize();
    g_sim = &sim;

    polyscope::init();
    polyscope::options::programName     = "MPM Fluid – Phase 2";
    polyscope::view::upDir              = polyscope::UpDir::YUp;
    polyscope::view::style              = polyscope::NavigateStyle::Planar;
    polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;

    sim.registerPolyscope();

    polyscope::view::lookAt(
        {5.0, 3.0, 20.0},
        {5.0, 3.0,  0.0},
        {0.0, 1.0,  0.0}
    );

    polyscope::state::userCallback = uiCallback;
    polyscope::show();
    return 0;
}
