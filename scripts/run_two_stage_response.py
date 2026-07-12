"""Two-stage country temperature-sensitivity diagnostic.

Stage 1 fits each country's own linear temperature sensitivity β₁ᵢ; Stage 2 asks what
explains β₁ᵢ across countries (income vs mean temperature). Standalone — separate from the
pooled "approaches" pipeline. See src/two_stage_response.py for the model.

Example:
    python scripts/run_two_stage_response.py
    python scripts/run_two_stage_response.py --min-years 25 --output-dir data/output/two_stage
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import load_data, load_data_from_csv
from src.fitting import fit_ApproachQJ_conjoined
from src.two_stage_response import fit_country_temperature_slopes, explain_country_slopes


def _print_stage1(slopes):
    b, se = slopes.beta1, slopes.se
    tag = "year-means removed" if slopes.remove_year_means else "raw (no year means)"
    sig = np.abs(b / se) > 1.96
    print(f"\nStage 1 [{tag}]: {len(b)} countries retained")
    print(f"  β₁ᵢ: median {np.median(b): .5f}   IQR [{np.percentile(b,25): .5f}, {np.percentile(b,75): .5f}]"
          f"   mean {np.mean(b): .5f}")
    print(f"  share β₁ᵢ > 0: {np.mean(b > 0)*100:.0f}%    share individually significant (|t|>1.96): {np.mean(sig)*100:.0f}%")


def _print_stage2(res, qj_h2):
    tag = "precision-weighted" if res['weighted'] else "unweighted"
    print(f"\nStage 2 [{tag}, n={res['n_countries']}]   "
          f"raw corr(β₁, logGDP)={res['corr_income']: .3f}   corr(β₁, meanT)={res['corr_meanT']: .3f}")
    for fit in ('income_only', 'temp_only', 'both'):
        terms = res['terms'][fit]
        parts = []
        for name, c, s in zip(terms, res[fit]['coef'], res[fit]['se']):
            if name == 'const':
                continue
            parts.append(f"{name}: {c: .3e} ± {s:.1e}  (t={c/s: .1f})")
        print(f"  {fit:<12} R²={res[fit]['r_squared']: .3f}   " + "   ".join(parts))
    # Sanity: β̂₁-vs-meanT slope should ≈ 2·β₂ from pooled QJ
    d1 = res['temp_only']['coef'][1]
    both_d = res['both']['coef'][2]
    print(f"  sanity: meanT slope (temp-only) {d1: .3e}  vs  2·β₂(QJ) {2*qj_h2: .3e}   "
          f"[mechanical-slope check]")
    print(f"  HEADLINE: income slope beyond meanT (both) = {res['both']['coef'][1]: .3e} "
          f"± {res['both']['se'][1]:.1e}  (t={res['both']['coef'][1]/res['both']['se'][1]: .1f})")


def _scatter(slopes, res, out_path):
    # Point size ∝ precision (1/se): larger = more precisely estimated slope.
    inv = 1.0 / slopes.se
    size = 12 + 140 * (inv - inv.min()) / (inv.max() - inv.min())
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    panels = (
        (axes[0], slopes.mean_logGDP, 'mean log(pcGDP)', 'income_only',
         slopes.mean_T, 'mean T (°C)', 'coolwarm'),
        (axes[1], slopes.mean_T, 'mean temperature (°C)', 'temp_only',
         slopes.mean_logGDP, 'mean log(pcGDP)', 'viridis'),
    )
    for ax, x, xlabel, key, cvar, clabel, cmap in panels:
        sc = ax.scatter(x, slopes.beta1, c=cvar, s=size, cmap=cmap,
                        alpha=0.85, edgecolors='k', linewidths=0.3)
        fig.colorbar(sc, ax=ax, label=clabel, fraction=0.046, pad=0.04)
        xc = x - x.mean()
        xs = np.linspace(xc.min(), xc.max(), 100)
        c0, c1 = res[key]['coef']
        t = c1 / res[key]['se'][1]
        ax.plot(xs + x.mean(), c0 + c1 * xs, color='crimson', lw=2.2,
                label=f'WLS slope {c1:.2e}  (t={t:.1f})')
        ax.axhline(0, color='gray', lw=0.8, ls=':')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('country temperature sensitivity  β̂₁ᵢ', fontsize=12)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Stage-1 country temperature slopes β̂₁ᵢ  (point size ∝ precision; '
                 'color = the confounded variable)', fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.savefig(str(out_path).replace('.pdf', '.png'), dpi=130)
    plt.close()


def _scatter_plane(slopes, out_path):
    """β̂₁ᵢ in the income–temperature plane: the rich⟺cool confound made visual."""
    b = slopes.beta1
    inv = 1.0 / slopes.se
    size = 15 + 150 * (inv - inv.min()) / (inv.max() - inv.min())
    vmax = float(np.percentile(np.abs(b), 90))       # symmetric so white = 0 (diverging)
    r = float(np.corrcoef(slopes.mean_logGDP, slopes.mean_T)[0, 1])
    fig, ax = plt.subplots(figsize=(10, 7.5))
    sc = ax.scatter(slopes.mean_logGDP, slopes.mean_T, c=b, s=size, cmap='RdBu',
                    vmin=-vmax, vmax=vmax, alpha=0.9, edgecolors='k', linewidths=0.3)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04,
                 label='temperature sensitivity β̂₁ᵢ  (red = GDP loss, blue = GDP gain)')
    ax.set_xlabel('mean log(pcGDP)', fontsize=12)
    ax.set_ylabel('mean temperature (°C)', fontsize=12)
    ax.set_title('Country temperature sensitivity in the income–temperature plane\n'
                 f'(rich⟺cool confound: corr = {r:.2f}; point size ∝ precision)', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.savefig(str(out_path).replace('.pdf', '.png'), dpi=130)
    plt.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Two-stage country temperature-sensitivity diagnostic")
    parser.add_argument("--use-csv", default="data/input/Maddison_CRU_dataset.csv")
    parser.add_argument("--min-years", type=int, default=20)
    parser.add_argument("--output-dir", default="data/output/two_stage")
    args = parser.parse_args(argv)

    csv_path = Path(args.use_csv).expanduser()
    data = load_data_from_csv(str(csv_path)) if csv_path.exists() else load_data()
    qj_h2 = fit_ApproachQJ_conjoined(data).h2  # pooled quadratic curvature, for the sanity check
    print(f"Pooled QJ curvature β₂ = {qj_h2: .3e}   (β̂₁ᵢ vs meanT slope should ≈ 2·β₂ = {2*qj_h2: .3e})")

    primary = fit_country_temperature_slopes(data, remove_year_means=True, min_years=args.min_years)
    raw = fit_country_temperature_slopes(data, remove_year_means=False, min_years=args.min_years)
    _print_stage1(primary)
    _print_stage1(raw)

    res_w = explain_country_slopes(primary, weighted=True)
    res_u = explain_country_slopes(primary, weighted=False)
    print("\n" + "=" * 78 + "\nStage 2 (year-means-removed slopes)\n" + "=" * 78)
    _print_stage2(res_w, qj_h2)
    _print_stage2(res_u, qj_h2)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        'iso': primary.iso, 'beta1': primary.beta1, 'se': primary.se,
        'n_obs': primary.n_obs, 'r_squared': primary.r_squared,
        'mean_logGDP': primary.mean_logGDP, 'mean_T': primary.mean_T,
        'growth_vol': primary.growth_vol,
    }).to_csv(out_dir / 'country_slopes.csv', index=False)
    _scatter(primary, res_w, out_dir / 'country_slopes.pdf')
    _scatter_plane(primary, out_dir / 'slope_in_gdp_temp_plane.pdf')
    print(f"\nSaved: {out_dir / 'country_slopes.csv'}, country_slopes.pdf/png, "
          f"slope_in_gdp_temp_plane.pdf/png")


if __name__ == "__main__":
    main()
