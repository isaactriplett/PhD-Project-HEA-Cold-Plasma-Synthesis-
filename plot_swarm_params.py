"""
plot_swarm_params.py
Six-panel electron swarm parameter figure from Ar_swarm_master.csv.

Derived quantities
------------------
  vd       [m/s]    = muN * EN_Td * 1e-21          (drift velocity)
  eps_k    [eV]     = DN / muN                      (characteristic energy = D/mu)
  k_iz     [m^3/s]  = alphaN_ion_m2 * vd            (ionisation rate coefficient)

Columns in CSV: EN_Td, Emean_eV, muN, DN, alphaN_ion_m2, source
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── load ──────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
df = pd.read_csv(r"C:\repos\PhD-Project-HEA-Cold-Plasma-Synthesis-\METHES\Ar\Ar_swarm_master.csv")

bolsig = df[df["source"] == "BOLSIG"].copy()
methes = df[df["source"] == "METHES"].copy()

STITCH_TD = 10.0   # Td — visual stitch boundary

def derive(d):
    d = d.copy()
    d["vd"]    = d["muN"] * d["EN_Td"] * 1e-21          # m/s
    d["eps_k"] = d["DN"]  / d["muN"]                     # eV (D/mu ratio)
    # k_iz only meaningful where ionisation is non-negligible
    d["k_iz"]  = d["alphaN_ion_m2"] * d["vd"]            # m^3/s
    d.loc[d["alphaN_ion_m2"] <= 0, "k_iz"] = np.nan
    return d

bolsig = derive(bolsig)
methes = derive(methes)

# ── figure layout ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
axes = axes.flatten()

BLUE   = "#2166ac"
RED    = "#d6604d"
STITCH = "#444444"

panels = [
    # (ax_idx, y_col,          y_label,                              y_log,  x_log)
    (0, "Emean_eV",      r"Mean energy  $\langle\varepsilon\rangle$  (eV)",  False, True),
    (1, "vd",            r"Drift velocity  $v_d$  (m s$^{-1}$)",            False, True),
    (2, "DN",            r"Reduced diffusion  $ND_T$  (m$^{-1}$ s$^{-1}$)", False, True),
    (3, "alphaN_ion_m2", r"Reduced ionisation  $\alpha/N$  (m$^{2}$)",       True,  True),
    (4, "k_iz",          r"Ionisation rate coeff.  $k_{iz}$  (m$^{3}$ s$^{-1}$)", True, True),
    (5, "eps_k",         r"Characteristic energy  $\varepsilon_k = D/\mu$  (eV)", False, True),
]

panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

for ax_idx, col, ylabel, y_log, x_log in panels:
    ax = axes[ax_idx]

    b_y = bolsig[col].values
    m_y = methes[col].values

    # mask NaN / zero for log axes
    mask_b = np.isfinite(b_y) & (b_y > 0 if y_log else np.ones(len(b_y), bool))
    mask_m = np.isfinite(m_y) & (m_y > 0 if y_log else np.ones(len(m_y), bool))

    ax.plot(bolsig["EN_Td"].values[mask_b], b_y[mask_b],
            color=RED,  lw=1.8, label="BOLSIG+")
    ax.plot(methes["EN_Td"].values[mask_m], m_y[mask_m],
            color=BLUE, lw=1.8, label="METHES")

    # stitch marker
    ax.axvline(STITCH_TD, color=STITCH, lw=1.1, ls="--", zorder=3)
    ax.axvspan(STITCH_TD * 0.7, STITCH_TD * 1.4,
               alpha=0.07, color=STITCH, zorder=0)

    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")

    ax.set_xlabel(r"$E/N$  (Td)", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)

    # panel label
    ax.text(0.04, 0.96, panel_labels[ax_idx],
            transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="top", ha="left")

# shared legend in top-right panel
axes[2].legend(fontsize=8, loc="upper right",
               framealpha=0.9, edgecolor="0.7")
# annotate stitch on one panel
axes[0].text(STITCH_TD * 1.05, axes[0].get_ylim()[0] * 1.02,
             "10 Td\nstitch", fontsize=7, color=STITCH,
             va="bottom", ha="left")

fig.suptitle(
    "Argon electron swarm parameters  |  BOLSIG+ (low field) + METHES (≥ 10 Td)",
    fontsize=11, fontweight="bold", y=1.01
)
fig.tight_layout()

out_png = HERE / "Ar_swarm_6panel.png"
out_pdf = HERE / "Ar_swarm_6panel.pdf"
fig.savefig(out_png, dpi=200, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
print(f"Saved:\n  {out_png}\n  {out_pdf}")
plt.show()
