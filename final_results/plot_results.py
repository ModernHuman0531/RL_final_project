"""
Final Results Visualization — 3x3 SUMO-RL Traffic Signal Control
=================================================================
Run this script inside the Docker container from /workspace:

    python3 final_results/plot_results.py

Outputs are saved to final_results/.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

OUT_DIR = "final_results"
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# HYPERPARAMETERS
# =============================================================================
#
# Original PPO  (results/ppo_1x1_20260605_155109_103036/)
#   --learning-rate   3e-4   default — relatively high, causes instability
#   --entropy-coef    0.01   default — low exploration, policy converges early
#   --update-epochs   3      default — few gradient updates per episode
#   --batch-size      64
#   --gamma           0.99
#   --gae-lambda      0.95
#   --ppo-clip-eps    0.2
#   --hidden-dim      128
#   --sim-end-time    3600s × 100 episodes
#
# Tuned PPO  (results/ppo_3x3_B1_tuned_hyperparams/)
#   --learning-rate   1e-4   reduced — more stable gradient steps
#   --entropy-coef    0.05   increased — more exploration, avoids early plateau
#   --update-epochs   5      increased — squeezes more learning per episode
#   --batch-size      64     (unchanged)
#   --gamma           0.99   (unchanged)
#   --gae-lambda      0.95   (unchanged)
#   --ppo-clip-eps    0.2    (unchanged)
#   --hidden-dim      128    (unchanged)
#   --sim-end-time    3600s × 100 episodes
#
# agent_final.pt vs best_agent.pt
#   agent_final.pt  — model saved after the LAST episode (ep100).
#                     Not necessarily the best policy found during training.
#   best_agent.pt   — model saved at the episode with the highest deterministic
#                     eval reward during training.
#                     Original PPO: best at ep60 (eval reward −46,554).
#                     Tuned PPO:    best at ep80 (eval reward −47,004).
#                     Always use best_agent.pt for fair evaluation.
# =============================================================================

# =============================================================================
# DATA
# =============================================================================

agents_short = [
    "PPO orig\n(best, ep60)",
    "PPO tuned\n(best, ep80)",
    "PPO tuned\n(final, ep100)",
    "DQN-AR",
    "SPRe+",
    "Max Pressure",
    "SOTL",
    "Fixed-Time",
]

agents_table = [
    "PPO original — best_agent (ep60)",
    "PPO tuned   — best_agent (ep80)",
    "PPO tuned   — agent_final (ep100)",
    "DQN-AR",
    "SPRe+ (DQN policy)",
    "Max Pressure",
    "SOTL (kappa=5)",
    "Fixed-Time (30s cycle)",
]

avg_wait       = [1291,  1304,  1350,  1351,  1351,  1394,  1406,  1555]
avg_queue      = [78.67, 79.78, 80.48, 79.61, 79.61, 80.00, 80.65, 79.00]
violation_rate = [0.000, 0.000, 0.000, 0.000, 0.000, 0.565, 0.710, 0.700]
reward         = [-4649048, -4693935, -4860739, -4862170, -4862170,
                  -5100000, -5189343, -5700000]

COLORS = [
    "#1565C0",  # PPO original best   — dark blue
    "#42A5F5",  # PPO tuned best      — medium blue
    "#90CAF9",  # PPO tuned final     — light blue
    "#2E7D32",  # DQN-AR              — dark green
    "#81C784",  # SPRe+               — light green
    "#E65100",  # Max Pressure        — orange
    "#B71C1C",  # SOTL               — dark red
    "#6A1B9A",  # Fixed-Time          — purple
]

n = len(agents_short)
y = np.arange(n)


# =============================================================================
# FIGURE 1 — Three-metric bar chart comparison
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.suptitle(
    "3×3 SUMO-RL Traffic Signal Control — Agent Comparison\n"
    "Scenario: 3×3 grid, intersection B1, 5 evaluation episodes × 3600 s",
    fontsize=13, fontweight="bold", y=1.01,
)

def hbar(ax, values, title, xlabel, color_fn=None, fmt="{:.0f}", invert=False):
    bars = ax.barh(
        y, values,
        color=[color_fn(v) if color_fn else COLORS[i] for i, v in enumerate(values)],
        edgecolor="white", linewidth=0.6, height=0.65,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(agents_short, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(0, color="grey", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + abs(bar.get_width()) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(val),
            va="center", ha="left", fontsize=8,
        )

# — Avg Wait
hbar(axes[0], avg_wait,
     "Average Vehicle Waiting Time", "Seconds",
     fmt="{:.0f} s")
axes[0].axvline(avg_wait[0], color=COLORS[0], linestyle="--", linewidth=1,
                alpha=0.5, label="PPO original best")

# — Violation Rate
hbar(axes[1], [v * 100 for v in violation_rate],
     "Constraint Violation Rate", "% of steps",
     fmt="{:.1f}%")
axes[1].axvline(0, color="#1565C0", linestyle="--", linewidth=1.2, alpha=0.6)

# — Reward
hbar(axes[2], reward,
     "Average Episode Reward", "Reward (higher = better)",
     fmt="{:,.0f}")

plt.tight_layout()
out1 = os.path.join(OUT_DIR, "metrics_comparison.png")
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out1}")


# =============================================================================
# FIGURE 2 — Summary table + interpretation
# =============================================================================

fig2 = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(3, 1, figure=fig2, height_ratios=[0.7, 3, 1.4],
                       hspace=0.05)

# ---- Header text (hyperparameter info) ----
ax_header = fig2.add_subplot(gs[0])
ax_header.axis("off")
header_text = (
    "Hyperparameters\n"
    "Original PPO:  lr=3e-4  |  entropy_coef=0.01  |  update_epochs=3  "
    "|  batch=64  |  γ=0.99  |  100 episodes × 3600 s\n"
    "Tuned PPO:     lr=1e-4  |  entropy_coef=0.05  |  update_epochs=5  "
    "|  batch=64  |  γ=0.99  |  100 episodes × 3600 s\n\n"
    "best_agent.pt = checkpoint with highest deterministic eval reward during training  "
    "(original: ep60 | tuned: ep80)\n"
    "agent_final.pt = model saved after the last training episode (ep100) — "
    "not necessarily the best policy"
)
ax_header.text(0.01, 0.95, header_text, transform=ax_header.transAxes,
               fontsize=8.5, va="top", ha="left",
               fontfamily="monospace",
               bbox=dict(boxstyle="round,pad=0.5", facecolor="#F3F4F6",
                         edgecolor="#CCCCCC"))

# ---- Table ----
ax_table = fig2.add_subplot(gs[1])
ax_table.axis("off")

col_labels = ["Agent", "Avg Wait (s)", "Avg Queue", "Violation Rate", "Reward"]
table_data = [
    [agents_table[i],
     f"{avg_wait[i]:,}",
     f"{avg_queue[i]:.2f}",
     f"{violation_rate[i]*100:.1f}%",
     f"{reward[i]:,.0f}"]
    for i in range(n)
]

tbl = ax_table.table(
    cellText=table_data,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1, 2.0)

# Style header row
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor("#1565C0")
    tbl[0, j].set_text_props(color="white", fontweight="bold")

# Style data rows
for i in range(n):
    row_color = COLORS[i] + "33"  # 20% opacity hex
    text_color = "#111111"
    for j in range(len(col_labels)):
        tbl[i + 1, j].set_facecolor(row_color)
        tbl[i + 1, j].set_text_props(color=text_color)
    # Bold agent name
    tbl[i + 1, 0].set_text_props(fontweight="bold", color=text_color)

# Highlight best value per column with a border
best_rows = {
    1: np.argmin(avg_wait),
    2: np.argmin(avg_queue),
    3: np.argmin(violation_rate),
    4: np.argmax(reward),
}
for col, row in best_rows.items():
    cell = tbl[row + 1, col]
    cell.set_edgecolor("#FFD700")
    cell.set_linewidth(2.5)

ax_table.set_title(
    "Final Evaluation Results — 3×3 Grid, Intersection B1, 5 Episodes",
    fontsize=12, fontweight="bold", pad=10,
)

# ---- Interpretation text ----
ax_footer = fig2.add_subplot(gs[2])
ax_footer.axis("off")
interpretation = (
    "Interpretation\n\n"
    "• Action-Constrained PPO achieves the lowest average vehicle waiting time (1291 s) and the smallest queue (78.67 vehicles) "
    "with zero constraint violations across all evaluated episodes.\n"
    "• The tuned PPO (best_agent, ep80) is close behind at 1304 s — the hyperparameter changes (lower lr, higher entropy) "
    "prevented early convergence but required more episodes to match the original.\n"
    "• DQN-AR and SPRe+ achieve zero violations and competitive wait times (1351 s) but do not outperform PPO, "
    "likely because the DQN policy was trained for only 100 short episodes.\n"
    "• All rule-based baselines (Max Pressure, SOTL, Fixed-Time) incur 55–71% violation rates, "
    "meaning they regularly select unsafe signal phases under the pedestrian safety constraints.\n"
    "• The key advantage of the learned agents (PPO, DQN-AR, SPRe+) is zero violations, "
    "demonstrating that constraint-aware training successfully enforces pedestrian safety.\n"
    "• Both PPO variants used only 100 training episodes. Further training would likely "
    "widen the gap over baselines and allow the tuned variant to surpass the original."
)
ax_footer.text(0.01, 0.98, interpretation, transform=ax_footer.transAxes,
               fontsize=9, va="top", ha="left",
               bbox=dict(boxstyle="round,pad=0.6", facecolor="#FFF8E1",
                         edgecolor="#F9A825"))

out2 = os.path.join(OUT_DIR, "summary_table.png")
fig2.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {out2}")


# =============================================================================
# FIGURE 3 — Waiting time vs violation rate scatter
# =============================================================================

fig3, ax = plt.subplots(figsize=(10, 6))

for i, (w, v, ag) in enumerate(zip(avg_wait, violation_rate, agents_table)):
    ax.scatter(w, v * 100, color=COLORS[i], s=180, zorder=3,
               edgecolors="white", linewidths=1.2)
    offset_x = 8 if i < 5 else -8
    ha = "left" if i < 5 else "right"
    ax.annotate(
        agents_table[i].split("—")[0].strip(),
        (w, v * 100),
        xytext=(offset_x, 5), textcoords="offset points",
        fontsize=8, ha=ha, color=COLORS[i], fontweight="bold",
    )

ax.set_xlabel("Average Vehicle Waiting Time (s)", fontsize=11)
ax.set_ylabel("Constraint Violation Rate (%)", fontsize=11)
ax.set_title(
    "Efficiency vs. Safety Trade-off\n"
    "Lower waiting time + lower violation rate = better",
    fontsize=12, fontweight="bold",
)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(1220, 1640)
ax.set_ylim(-5, 82)

# Ideal quadrant annotation
ax.axhline(5, color="grey", linestyle=":", linewidth=0.8, alpha=0.5)
ax.axvline(1340, color="grey", linestyle=":", linewidth=0.8, alpha=0.5)
ax.text(1225, 1, "← ideal region", fontsize=8, color="grey")

# Legend
legend_handles = [
    mpatches.Patch(color=COLORS[0], label="PPO original (best)"),
    mpatches.Patch(color=COLORS[1], label="PPO tuned (best)"),
    mpatches.Patch(color=COLORS[2], label="PPO tuned (final)"),
    mpatches.Patch(color=COLORS[3], label="DQN-AR"),
    mpatches.Patch(color=COLORS[4], label="SPRe+"),
    mpatches.Patch(color=COLORS[5], label="Max Pressure"),
    mpatches.Patch(color=COLORS[6], label="SOTL"),
    mpatches.Patch(color=COLORS[7], label="Fixed-Time"),
]
ax.legend(handles=legend_handles, fontsize=8, loc="upper left",
          framealpha=0.9, ncol=2)

out3 = os.path.join(OUT_DIR, "efficiency_vs_safety.png")
fig3.savefig(out3, dpi=150, bbox_inches="tight")
plt.close(fig3)
print(f"Saved: {out3}")

print("\nDone. All figures saved to:", OUT_DIR)
