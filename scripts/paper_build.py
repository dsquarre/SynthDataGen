"""Regenerate every figure, table and number macro used by paper/main.tex, then compile it.

    python scripts/paper_build.py --e0 RUN_1TO1 --e0-compare RUN_11TO1 --e1 E1_DIR [--no-pdf]

Figures are drawn without embedded captions (captions live in main.tex) and cropped to their content.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

from PIL import Image, ImageChops

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(PROJECT, 'paper')
FIGS = os.path.join(PAPER, 'figures')


def crop(src, dst, pad=12):
    im = Image.open(src).convert('RGB')
    bg = Image.new('RGB', im.size, (255, 255, 255))
    box = ImageChops.difference(im, bg).getbbox()
    if box:
        l, t, r, b = box
        im = im.crop((max(0, l - pad), max(0, t - pad), min(im.width, r + pad), min(im.height, b + pad)))
    im.save(dst)


def run(cmd):
    print('+', ' '.join(cmd))
    subprocess.run(cmd, check=True, cwd=PROJECT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--e0', required=True)
    ap.add_argument('--e0-compare', required=True)
    ap.add_argument('--e1', required=True)
    ap.add_argument('--no-pdf', action='store_true')
    a = ap.parse_args()
    os.makedirs(FIGS, exist_ok=True)
    py = sys.executable
    with tempfile.TemporaryDirectory() as tmp:
        run([py, 'scripts/make_deck_figures.py', a.e0, tmp, '--ratio-label', '1:1',
             '--compare-run', a.e0_compare, '--compare-label', '11:1', '--no-caption'])
        crop(os.path.join(tmp, 'fig04_pca_iter.png'), os.path.join(FIGS, 'fig_pca_iter.png'))
        crop(os.path.join(tmp, 'fig13_ratio.png'), os.path.join(FIGS, 'fig_ratio.png'))
        run([py, 'scripts/make_flowchart.py', os.path.join(tmp, 'flow.png')])
        crop(os.path.join(tmp, 'flow.png'), os.path.join(FIGS, 'fig_flowchart.png'))
        run([py, 'scripts/e1_figures.py', os.path.join(a.e1, 'e1_all_summary.csv'), os.path.join(tmp, 'e1.png'),
             '--no-caption'])
        crop(os.path.join(tmp, 'e1.png'), os.path.join(FIGS, 'fig_e1.png'))
    run([py, 'scripts/racfst_series_check.py', '--out', os.path.join(PAPER, 'generated', 'series_numbers.tex')])
    run([py, 'scripts/paper_tables.py', '--e0', a.e0, '--e0-compare', a.e0_compare, '--e1', a.e1,
         '--out', os.path.join(PAPER, 'generated')])
    if not a.no_pdf:
        subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error', 'main.tex'],
                       check=True, cwd=PAPER, stdout=subprocess.DEVNULL)
        print('wrote', os.path.join(PAPER, 'main.pdf'))


if __name__ == '__main__':
    main()
