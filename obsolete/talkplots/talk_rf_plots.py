"""
Talk (dark-background) versions of the Random Forest feature-importance figures.

Produces six standalone PNGs -- Random Forest classification (quenching) and
regression (SFR) for each of the three galaxy samples (All Centrals,
High-Mass Satellites, Low-Mass Satellites) -- plus a seventh figure containing
only the legend.

Each panel carries the same detail as the publication six-panel figures
(bars with outlines + error bars, LaTeX feature labels, colour-coded
local/intrinsic/environmental grouping, an inset pie chart, and a train/test
metric annotation) but is restyled for a black background:

  * transparent figure/axes background,
  * every element that is black in the publication figures is drawn in white
    (title, tick/axis labels, bar outlines, error bars, spines, grid, pie
    outlines, leader lines, metric box, pie category labels),
  * the ONLY black text kept is the percentage on top of each pie wedge.

The legend is a separate frameless figure with white text on a transparent
background.

Run from anywhere -- data paths are resolved relative to this file.
"""

import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # noqa: F401

# --------------------------------------------------------------------------- #
# Data locations (relative to the repo root, i.e. the parent of talkplots/)
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = HERE  # PNGs are written next to this script

RF_CLF = os.path.join(
    ROOT, "classification_revised", "rf_classification_annuli",
    "rf_classification_outputs",
)
RF_REG = os.path.join(
    ROOT, "regression_revised", "rf_regression_annuli", "rf_regression_outputs",
)

# --------------------------------------------------------------------------- #
# Feature groupings, colours and LaTeX labels (identical to the pub figures)
# --------------------------------------------------------------------------- #
local_features = {'bin_radii', 'avg_s_mass'}
intrinsic_global_features = {'s_mass', 'BHMass', 'bulge', 'BH_AGN'}
environmental_global_features = {'Mhalo', 'overdens'}
local_color = '#EEDD88'
intrinsic_global_color = '#FFAABB'
environmental_global_color = '#99DDFF'

latex_labels = {
    'bin_radii':   r'$r/R_e$',
    'avg_s_mass':  r'$\Sigma_{{*}}$',
    's_mass':      r'$M_{{*}}$',
    'BHMass':      r'$M_{{BH}}$',
    'Mhalo':       r'$M_{{h}}$',
    'bulge':       r'$F(G, M_{{20}})$',
    'overdens':    r'$\delta_{{5}}$',
    'BH_AGN':      r'$\Sigma E_{{AGN}}$',
}

# White for everything that is black in the publication figures.
WHITE = 'white'


def _apply_dark_style():
    """Matplotlib rcParams so black elements render white on a transparent bg."""
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        # Everything that used to be black -> white
        "text.color": WHITE,
        "axes.labelcolor": WHITE,
        "axes.edgecolor": WHITE,
        "axes.titlecolor": WHITE,
        "xtick.color": WHITE,
        "ytick.color": WHITE,
        # Transparent backgrounds
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.transparent": True,
    })


def single_panel(imp, metric, title, metric_type, save_path,
                 figsize=(7.5, 7.5), pie_xy=(0.34, 0.18),
                 mid_threshold=5, under_threshold=2):
    """Dark-background, single-sample feature-importance figure.

    Parameters
    ----------
    imp : pd.DataFrame
        Columns 'Feature', 'Mean Importance', 'Std Dev'.
    metric : pd.DataFrame
        Single-row frame with 'Train <metric>' / 'Test <metric>'.
    title : str
        Galaxy-sample title (e.g. 'All Centrals').
    metric_type : str
        'AUC' (classification) or 'MSE' (regression).
    save_path : str
        Destination PNG path.
    figsize, pie_xy, mid_threshold, under_threshold
        Layout knobs, matching the publication figure semantics.
    """
    _apply_dark_style()
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.4)
    # sns.set_style resets some colours -- re-apply the dark rcParams.
    _apply_dark_style()

    df = imp.sort_values('Mean Importance', ascending=False).copy()
    df['orig_feature'] = df['Feature']
    df['Feature'] = df['Feature'].map(lambda x: latex_labels.get(x, x))
    max_imp = (df['Mean Importance'] + df['Std Dev']).max() * 1.1

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    colors = [local_color if feat in local_features
              else intrinsic_global_color if feat in intrinsic_global_features
              else environmental_global_color for feat in df['orig_feature']]

    ax.bar(
        df['Feature'], df['Mean Importance'], yerr=df['Std Dev'],
        capsize=5, color=colors, edgecolor=WHITE, linewidth=0.8, alpha=1.0,
        error_kw={'ecolor': WHITE, 'capthick': 1.8, 'elinewidth': 1.8},
    )

    # White border around the subplot
    for spine in ax.spines.values():
        spine.set_edgecolor(WHITE)
        spine.set_linewidth(0.8)

    ax.set_title(title, fontsize=22, color=WHITE)
    ax.set_ylim(0, max_imp)
    ax.grid(False)
    ax.tick_params(axis='x', labelrotation=45, labelsize=17, colors=WHITE)
    ax.tick_params(axis='y', labelsize=17, colors=WHITE)
    ax.set_ylabel(r'Relative Importance', fontsize=20, color=WHITE)

    # ------------------------------------------------------------------ #
    # Inset pie chart
    # ------------------------------------------------------------------ #
    x0, y0 = pie_xy
    axins = ax.inset_axes([x0, y0, 0.6, 0.6], transform=ax.transAxes)
    axins.patch.set_alpha(0.0)
    local_sum = df.loc[df['orig_feature'].isin(local_features), 'Mean Importance'].sum()
    intrinsic_global_sum = df.loc[df['orig_feature'].isin(intrinsic_global_features), 'Mean Importance'].sum()
    environmental_global_sum = df.loc[df['orig_feature'].isin(environmental_global_features), 'Mean Importance'].sum()
    values = [local_sum, intrinsic_global_sum, environmental_global_sum]
    labels = ['Local', 'Intr.', 'Env.']
    wedges, texts, autotexts = axins.pie(
        values,
        colors=[local_color, intrinsic_global_color, environmental_global_color],
        startangle=90,
        wedgeprops={'alpha': 1.0, 'linewidth': 0.5, 'edgecolor': WHITE},
        labels=labels,
        autopct=lambda pct: f"{pct:.0f}%",
        textprops={'fontsize': 15},
    )

    total = sum(values)
    small_labels = []

    for text, autotext, value in zip(texts, autotexts, values):
        pct = value / total * 100 if total > 0 else 0
        text.set_fontsize(15)
        text.set_color(WHITE)          # category label -> white
        autotext.set_fontsize(15)
        autotext.set_color('black')    # percentage stays BLACK (per request)

        # Very small slices: list above the pie to avoid label collisions
        if pct < under_threshold:
            text.set_visible(False)
            autotext.set_visible(False)
            small_labels.append(f"{text.get_text()} ({pct:.0f}%)")
        # Small slices: move the percentage label outside with a leader line
        elif pct < mid_threshold:
            autotext.set_visible(False)
            x, y = text.get_position()
            text.set_position((1.2 * x, 1.2 * y))
            text.set_text(f"{text.get_text()} ({pct:.0f}%)")
            axins.plot([0.9 * x, 1.1 * x], [0.9 * y, 1.1 * y], color=WHITE, lw=0.7, zorder=0)
        # Standard case: label outside, percentage inside
        else:
            x, y = text.get_position()
            text.set_position((1.2 * x, 1.2 * y))
            axins.plot([0.9 * x, 1.1 * x], [0.9 * y, 1.1 * y], color=WHITE, lw=0.7, zorder=0)

    # List the very small slice labels above the pie, in the gap between the
    # pie top and the metric box.
    if small_labels:
        label_str = "\n".join(small_labels)
        ax.text(pie_xy[0] + 0.30, 0.84, label_str, transform=ax.transAxes,
                ha='center', va='top', fontsize=15, color=WHITE)

    axins.set_aspect('equal')

    # ------------------------------------------------------------------ #
    # Train/test metric annotation (white text, white-outlined transparent box)
    # ------------------------------------------------------------------ #
    mdf = metric.copy()
    mdf.columns = mdf.columns.str.strip()
    train = mdf[f'Train {metric_type}'].iloc[0]
    test = mdf[f'Test {metric_type}'].iloc[0]
    textstr = f"Train {metric_type} = {train:.2f}\nTest {metric_type} = {test:.2f}"
    ax.text(0.56, 0.95, textstr, transform=ax.transAxes,
            fontsize=15, verticalalignment='top', color=WHITE,
            bbox=dict(boxstyle='round', facecolor='none', edgecolor=WHITE, alpha=0.9))

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)
    print(f"wrote {save_path}")


def legend_figure(save_path, figsize=(4, 2)):
    """Standalone frameless legend, white text on a transparent background."""
    _apply_dark_style()
    handles = [
        plt.Line2D([0], [0], marker='s', color=local_color, lw=0, markersize=14, label='Local'),
        plt.Line2D([0], [0], marker='s', color=intrinsic_global_color, lw=0, markersize=14, label='Intrinsic Global'),
        plt.Line2D([0], [0], marker='s', color=environmental_global_color, lw=0, markersize=14, label='Environmental Global'),
    ]
    fig = plt.figure(figsize=figsize)
    fig.patch.set_alpha(0.0)
    leg = fig.legend(handles=handles, loc='center', frameon=False, fontsize=18)
    for text in leg.get_texts():
        text.set_color(WHITE)
    fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)
    print(f"wrote {save_path}")


def main():
    # (title, csv basename, sample key) for each galaxy sample
    samples = [
        ('All Centrals',          'all_centrals'),
        ('High-Mass Satellites',  'hms'),
        ('Low-Mass Satellites',   'lms'),
    ]

    # ----- Classification (quenching), metric = AUC ----- #
    for title, key in samples:
        imp = pd.read_csv(os.path.join(RF_CLF, f"{key}_rf_quenching.csv"))
        metric = pd.read_csv(os.path.join(RF_CLF, f"{key}_rf_quenching_auc.csv"))
        single_panel(
            imp, metric, title, metric_type='AUC',
            save_path=os.path.join(OUT_DIR, f"rf_classification_{key}.png"),
            mid_threshold=5, under_threshold=2,
        )

    # ----- Regression (SFR), metric = MSE ----- #
    for title, key in samples:
        imp = pd.read_csv(os.path.join(RF_REG, f"{key}_rf_reg.csv"))
        metric = pd.read_csv(os.path.join(RF_REG, f"{key}_rf_reg_mse.csv"))
        single_panel(
            imp, metric, title, metric_type='MSE',
            save_path=os.path.join(OUT_DIR, f"rf_regression_{key}.png"),
            mid_threshold=7, under_threshold=2,
        )

    # ----- Standalone legend ----- #
    legend_figure(os.path.join(OUT_DIR, "legend.png"))


if __name__ == "__main__":
    main()
