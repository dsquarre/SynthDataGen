"""Simplified methodology flowchart for the deck (slide 5).

Drawn at the exact size of the slide's picture box (3.52 x 5.37 in) so text appears at its true point size.
Usage: python scripts/make_flowchart.py OUT.png
"""
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

W, H = 3.52, 5.37
FONT = 'Liberation Serif'
EDGE = '#2f4b7c'
FILL = '#e8eef7'
FILL_EVAL = '#eef5e9'
EDGE_EVAL = '#3d7a3d'

fig = plt.figure(figsize=(W, H), dpi=300)
ax = fig.add_axes((0, 0, 1, 1))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis('off')


def box(x, y, w, h, title, sub=None, fill=FILL, edge=EDGE, tsize=9.5, ssize=7.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.06',
                                fc=fill, ec=edge, lw=1.1))
    if sub:
        ax.text(x + w / 2, y + h * 0.66, title, ha='center', va='center', family=FONT, weight='bold', size=tsize)
        ax.text(x + w / 2, y + h * 0.3, sub, ha='center', va='center', family=FONT, size=ssize, linespacing=1.15)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha='center', va='center', family=FONT, weight='bold', size=tsize)


def arrow(y_from, y_to, x=W / 2):
    ax.annotate('', xy=(x, y_to), xytext=(x, y_from),
                arrowprops=dict(arrowstyle='-|>', color='#555555', lw=1.1, mutation_scale=9))


def stage_label(y, text):
    ax.text(0.06, y, text, ha='left', va='bottom', family=FONT, size=7, color='#555555', style='italic')


M = 0.1           # side margin
FW = W - 2 * M    # full box width

# 1. data
box(M, 4.72, FW, 0.55, 'Experimental data', '56 cyclic RACFST column tests\n(Xu et al., 2020)')
arrow(4.72, 4.50)

# 2. preprocessing
box(M, 3.92, FW, 0.58, 'Preprocessing', 'Ratio variables removed, rebuilt later\nSection shape completed (Random Forest)')
arrow(3.92, 3.72)

# 3. imputation branches
stage_label(3.66, 'Missing-data imputation (3 branches)')
bw = (FW - 2 * 0.06) / 3
for i, name in enumerate(['Iterative', 'KNN', 'Missing\nindicator']):
    box(M + i * (bw + 0.06), 3.12, bw, 0.5, name, tsize=8.5)
arrow(3.12, 2.92)

# 4. generators
stage_label(2.86, 'Synthetic data generation (5 models)')
box(M, 1.98, FW, 0.84, '', fill=FILL)
ax.text(W / 2, 2.62, 'Sequential CART  ·  Gaussian Copula', ha='center', va='center', family=FONT, weight='bold', size=8.5)
ax.text(W / 2, 2.43, 'CTGAN  ·  CopulaGAN  ·  TVAE', ha='center', va='center', family=FONT, weight='bold', size=8.5)
ax.text(W / 2, 2.15, 'Rows outside physical bounds are discarded and regenerated', ha='center', va='center', family=FONT, size=7.5)
arrow(1.98, 1.80)

# 5. evaluation
stage_label(1.74, 'Evaluation')
ew = (FW - 2 * 0.06) / 3
labels = [('Fidelity', 'Distributions,\ncorrelations'),
          ('Originality', 'Distance to\nreal records'),
          ('Utility', 'Pu prediction\nvs real-only')]
for i, (t, s) in enumerate(labels):
    box(M + i * (ew + 0.06), 0.84, ew, 0.86, t, s, fill=FILL_EVAL, edge=EDGE_EVAL, tsize=8.5, ssize=7)

ax.text(W / 2, 0.42, 'Utility: synthetic data generated inside each training\nfold only; tested on held-out real data (repeated 5-fold CV)',
        ha='center', va='center', family=FONT, size=7.5, color='#333333', linespacing=1.2)

fig.savefig(sys.argv[1], dpi=300)
print('wrote', sys.argv[1])
