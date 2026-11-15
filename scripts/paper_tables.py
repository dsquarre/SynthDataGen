"""Write LaTeX tables and number macros for the journal manuscript from run outputs.

    python scripts/paper_tables.py --e0 RUN_1TO1 --e0-compare RUN_11TO1 --e1 E1_DIR --out paper/generated

Outputs (all \\input by paper/main.tex):
  tab_fidelity.tex    fidelity/originality metrics per generator and imputation (E0, 1:1 run)
  tab_regression.tex  mean test MAE per setting, real-only vs 1:1 vs 11:1 (E0)
  tab_e1_counts.tex   E1 verdict counts per dataset and generator
  tab_e1_cells.tex    E1 median relative MAE change for every cell (appendix)
  numbers.tex         \\newcommand macros for every number quoted in the text
"""
import argparse
import os

import numpy as np
import pandas as pd

GEN_ORDER = ['cart', 'gaussian copula', 'ctgan', 'copulagan', 'tvae']
GEN_TEX = {'cart': 'Sequential CART', 'gaussian copula': 'Gaussian Copula', 'ctgan': 'CTGAN',
           'copulagan': 'CopulaGAN', 'tvae': 'TVAE'}
E1_GEN = {'CART': 'cart', 'GaussianCopula': 'gaussian copula', 'CTGAN': 'ctgan', 'CopulaGAN': 'copulagan', 'TVAE': 'tvae'}
IMP_ORDER = ['iter', 'knn', 'mi']
IMP_TEX = {'iter': 'Iterative', 'knn': 'KNN', 'mi': 'Missing indicator'}
DS_TEX = {'frp_confined_concrete': 'FRP-confined concrete', 'cfst_rectangular_hss': 'Rectangular HSS CFST',
          'cfst_circular_hss': 'Circular HSS CFST$^{\\dagger}$'}
DS_ORDER = list(DS_TEX)

macros = {}


def macro(name, value):
    assert name.isalpha(), name
    macros[name] = value


def gen_key(model_name):
    base = model_name.replace('_', ' ').rsplit(' ', 1)[0]
    return base


# ------------------------------------------------------------------ E0 fidelity
def fidelity_table(run, out):
    e = pd.read_excel(os.path.join(run, 'evaluation_report.xlsx'))
    e['gen'] = e.Model.map(gen_key)
    lines = [r'\begin{tabular}{@{}llrrrrrrr@{}}', r'\toprule',
             r'Imputation & Generator & Bound. & Corr. & Cov. & WD & FD ($\times10^3$) & DCR$_\text{mean}$ & Prop. \\',
             r'\midrule']
    for imp in IMP_ORDER:
        sub = e[e.Imputation == imp].set_index('gen')
        best = {'cov_score': sub.cov_score.idxmax(), 'corr_score': sub.corr_score.idxmax(),
                'wd_score': sub.wd_score.idxmin(), 'fd_score': sub.fd_score.idxmin()}
        for k, gen in enumerate(GEN_ORDER):
            r = sub.loc[gen]

            def fmt(col, text):
                return r'\textbf{' + text + '}' if best.get(col) == gen else text
            lines.append(' & '.join([
                IMP_TEX[imp] if k == 0 else '', GEN_TEX[gen], f'{r.boundary_score:.3f}',
                fmt('corr_score', f'{r.corr_score:.3f}'), fmt('cov_score', f'{r.cov_score:.3f}'),
                fmt('wd_score', f'{r.wd_score:.1f}'), fmt('fd_score', f'{r.fd_score / 1e3:,.0f}'),
                f'{r.DCR_mean:.2f}', f'{r.propensity_score:.3f}']) + r' \\')
        if imp != IMP_ORDER[-1]:
            lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(out, 'tab_fidelity.tex'), 'w').write('\n'.join(lines) + '\n')

    cart = e[e.gen == 'cart']
    macro('cartCovMin', f'{cart.cov_score.min():.2f}')
    macro('cartCovMax', f'{cart.cov_score.max():.2f}')
    gan = e[e.gen.isin(['ctgan', 'copulagan'])]
    macro('ganCovMin', f'{gan.cov_score.min():.2f}')
    macro('ganCovMax', f'{gan.cov_score.max():.2f}')
    macro('boundMin', f'{e.boundary_score.min():.3f}')
    macro('boundMax', f'{e.boundary_score.max():.3f}')
    macro('propMax', f'{e.propensity_score.max():.2f}')
    macro('dcrMinAll', f'{e.DCR_min.min():.2f}')
    near = e[e.gen.isin(['cart', 'tvae'])]
    far = e[~e.gen.isin(['cart', 'tvae'])]
    macro('dcrNearMin', f'{near.DCR_mean.min():.2f}')
    macro('dcrNearMax', f'{near.DCR_mean.max():.2f}')
    macro('dcrFarMin', f'{far.DCR_mean.min():.2f}')
    macro('dcrFarMax', f'{far.DCR_mean.max():.2f}')
    macro('mmdMin', f'{e.mmd_score.min():.4f}')
    macro('mmdMax', f'{e.mmd_score.max():.4f}')
    wins = {c: sum(e[e.Imputation == i].set_index('gen')[c].idxmax() == 'cart' for i in IMP_ORDER)
            for c in ['cov_score', 'corr_score']}
    wins.update({c: sum(e[e.Imputation == i].set_index('gen')[c].idxmin() == 'cart' for i in IMP_ORDER)
                 for c in ['wd_score', 'fd_score']})
    words = {0: 'none', 1: 'one', 2: 'two', 3: 'all three'}
    macro('cartCovWins', words[wins['cov_score']])
    macro('cartCorrWins', words[wins['corr_score']])
    macro('cartWdWins', words[wins['wd_score']])
    macro('cartFdWins', words[wins['fd_score']])
    return e


# ------------------------------------------------------------------ E0 regression
def _reg_summary(run):
    s = pd.read_excel(os.path.join(run, 'regression_report.xlsx'), 'summary')
    f = pd.read_excel(os.path.join(run, 'regression_report.xlsx'), 'fold_scores')
    s['gen'] = np.where(s.Evaluation.str.startswith('Real'), 'real', s.Synthetic_Model.map(gen_key))
    f['gen'] = np.where(f.Evaluation.str.startswith('Real'), 'real', f.Synthetic_Model.map(gen_key))
    s = s.drop_duplicates(['Imputation', 'gen', 'Regressor'])
    return s, f


def regression_table(run1, run11, out):
    s1, f1 = _reg_summary(run1)
    s11, f11 = _reg_summary(run11)
    cols = [(imp, reg) for imp in IMP_ORDER for reg in ['RandomForest', 'XGBoost']]
    get = lambda s, gen, imp, reg: s[(s.gen == gen) & (s.Imputation == imp) & (s.Regressor == reg)].mae_mean.iloc[0]
    lines = [r'\begin{tabular}{@{}lrrrrrr@{}}', r'\toprule',
             r' & \multicolumn{2}{c}{Iterative} & \multicolumn{2}{c}{KNN} & \multicolumn{2}{c}{Missing indicator} \\',
             r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}',
             r'Training data & RF & XGB & RF & XGB & RF & XGB \\', r'\midrule']
    real = {c: get(s1, 'real', *c) for c in cols}
    lines.append('Real only & ' + ' & '.join(f'{real[c]:.2f}' for c in cols) + r' \\')
    better = {'1': 0, '11': 0}
    deltas = {'1': [], '11': []}
    for label, s, key in [('1:1', s1, '1'), ('11:1', s11, '11')]:
        lines.append(r'\midrule')
        lines.append(r'\multicolumn{7}{@{}l}{\textit{Real + synthetic, ' + label + r'}} \\')
        for gen in GEN_ORDER:
            cells = []
            for c in cols:
                v = get(s, gen, *c)
                d = v - real[c]
                deltas[key].append((gen, c, d))
                if d < 0:
                    better[key] += 1
                cells.append((r'\textbf{' + f'{v:.2f}' + '}') if d < 0 else f'{v:.2f}')
            lines.append(r'\quad ' + GEN_TEX[gen] + ' & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(out, 'tab_regression.tex'), 'w').write('\n'.join(lines) + '\n')

    macro('regBetterOne', str(better['1']))
    macro('regBetterEleven', str(better['11']))
    macro('regSettings', str(len(cols) * len(GEN_ORDER)))
    neg1 = [d for d in deltas['1'] if d[2] < 0]
    macro('regBestGainOne', f'{-min(d[2] for d in deltas["1"]):.2f}' if neg1 else '0')
    macro('regBetterOneWho', '; '.join(f'{GEN_TEX[g]}, {IMP_TEX[c[0]]}, {"RF" if c[1] == "RandomForest" else "XGB"}'
                                       for g, c, _ in neg1) or 'none')
    nc1 = [d[2] for d in deltas['1'] if d[0] != 'cart']
    macro('regOtherMinOne', f'{min(nc1):+.2f}')
    macro('regOtherMaxOne', f'{max(nc1):+.2f}')
    macro('regElevenMin', f'{min(d[2] for d in deltas["11"]):+.2f}')
    macro('regElevenMax', f'{max(d[2] for d in deltas["11"]):+.2f}')
    gen_mean = {k: {g: np.mean([d[2] for d in deltas[k] if d[0] == g]) for g in GEN_ORDER} for k in deltas}
    macro('elevenWorseAll', 'yes' if all(gen_mean['11'][g] > gen_mean['1'][g] for g in GEN_ORDER) else 'no')
    # fold-level consistency of the settings where 1:1 lowered the mean MAE (descriptive; E0 is not pre-registered)
    from scipy.stats import wilcoxon
    fr = f1[f1.gen == 'real'].drop_duplicates(['Imputation', 'Regressor', 'Fold'])
    stats = []
    for gen, c, d in deltas['1']:
        a = f1[(f1.gen == gen) & (f1.Imputation == c[0]) & (f1.Regressor == c[1])].set_index('Fold').mae
        r = fr[(fr.Imputation == c[0]) & (fr.Regressor == c[1])].set_index('Fold').mae.reindex(a.index)
        stats.append(dict(gen=gen, d=d, pct=100 * d / real[c], wins=int((a < r).sum()), n=len(a),
                          p=wilcoxon(a, r).pvalue if (a != r).any() else 1.0))
    st = pd.DataFrame(stats)
    better, worse = st[st.d < 0], st[st.d > 0]
    if len(better):
        macro('regBetterOnePctMin', f'{-better.pct.max():.0f}')
        macro('regBetterOnePctMax', f'{-better.pct.min():.0f}')
        macro('regBetterOneWinsMin', str(better.wins.min()))
        macro('regBetterOneWinsMax', str(better.wins.max()))
        macro('regBetterOnePMin', f'{better.p.min():.2f}')
        macro('regBetterOneRegs', ' and '.join(sorted({'Random Forest' if c[1] == 'RandomForest' else 'XGBoost'
                                                        for g, c, d in deltas['1'] if d < 0})))
        macro('regBetterOneGens', ' and '.join(GEN_TEX[g] for g in GEN_ORDER if g in set(better.gen)))
    macro('regWorseOne', str(len(worse)))
    macro('regWorseOnePctMin', f'{worse.pct.min():.0f}')
    macro('regWorseOnePctMax', f'{worse.pct.max():.0f}')
    macro('regWorseOneKnMax', f'{worse.d.max():.2f}')
    macro('realMaeMin', f'{min(real.values()):.2f}')
    macro('realMaeMax', f'{max(real.values()):.2f}')
    r2 = s1[s1.gen == 'real'].r2_mean
    macro('realRtwoMin', f'{r2.min():.2f}')
    macro('realRtwoMax', f'{r2.max():.2f}')
    # worst real-only fold vs CART-augmented worst fold (iterative imputation)
    for reg, tag in [('RandomForest', 'RF'), ('XGBoost', 'XGB')]:
        wr = f1[(f1.gen == 'real') & (f1.Imputation == 'iter') & (f1.Regressor == reg)].mae.max()
        wc = f1[(f1.gen == 'cart') & (f1.Imputation == 'iter') & (f1.Regressor == reg)].mae.max()
        macro(f'worstReal{tag}', f'{wr:.1f}')
        macro(f'worstCart{tag}', f'{wc:.1f}')
    return s1, s11


# ------------------------------------------------------------------ E1
def e1_tables(e1dir, out):
    reg = pd.read_csv(os.path.join(e1dir, 'e1_registered_summary.csv'))
    exp = pd.read_csv(os.path.join(e1dir, 'e1x_cfst_circ_exploratory_summary.csv'))
    s = pd.concat([reg, exp], ignore_index=True)
    s['gen'] = s.generator.map(E1_GEN)

    lines = [r'\begin{tabular}{@{}llrrrr@{}}', r'\toprule',
             r'Dataset & Generator & Benefit & Harm & None & Median change (\%) \\', r'\midrule']
    for ds in DS_ORDER:
        sub = s[s.dataset == ds]
        for k, gen in enumerate(GEN_ORDER):
            g = sub[sub.gen == gen]
            v = g.verdict.value_counts()
            lines.append(' & '.join([DS_TEX[ds] if k == 0 else '', GEN_TEX[gen], str(v.get('Benefit', 0)),
                                     str(v.get('Harm', 0)), str(v.get('No detectable effect', 0)),
                                     f'{100 * g.median_d.min():+.1f} to {100 * g.median_d.max():+.1f}']) + r' \\')
        if ds != DS_ORDER[-1]:
            lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(out, 'tab_e1_counts.tex'), 'w').write('\n'.join(lines) + '\n')

    head = ' & '.join(GEN_TEX[g].replace('Sequential ', '').replace('Gaussian Copula', 'GC') for g in GEN_ORDER)
    lines = [r'\begin{tabular}{@{}lr' + 'r' * 10 + r'@{}}', r'\toprule',
             r' & & \multicolumn{5}{c}{XGBoost} & \multicolumn{5}{c}{Random Forest} \\',
             r'\cmidrule(lr){3-7}\cmidrule(lr){8-12}', r'Dataset & $n$ & ' + head + ' & ' + head + r' \\', r'\midrule']
    for ds in DS_ORDER:
        sub = s[s.dataset == ds]
        for k, n in enumerate(sorted(sub.n.unique())):
            cells = []
            for rg in ['XGBoost', 'RandomForest']:
                for gen in GEN_ORDER:
                    r = sub[(sub.n == n) & (sub.regressor == rg) & (sub.gen == gen)].iloc[0]
                    t = f'{100 * r.median_d:+.1f}'
                    if r.verdict == 'Harm':
                        t = r'\textbf{' + t + '}'
                    elif r.verdict == 'Benefit':
                        t = r'\underline{' + t + '}'
                    cells.append(t)
            lines.append((DS_TEX[ds] if k == 0 else '') + f' & {n} & ' + ' & '.join(cells) + r' \\')
        if ds != DS_ORDER[-1]:
            lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(out, 'tab_e1_cells.tex'), 'w').write('\n'.join(lines) + '\n')

    v = reg.verdict.value_counts()
    macro('eoneCells', str(len(reg)))
    macro('eoneBenefit', str(v.get('Benefit', 0)))
    macro('eoneHarm', str(v.get('Harm', 0)))
    macro('eoneNone', str(v.get('No detectable effect', 0)))
    for ds, tag in [('frp_confined_concrete', 'Frp'), ('cfst_rectangular_hss', 'Rect'), ('cfst_circular_hss', 'Circ')]:
        vv = s[s.dataset == ds].verdict.value_counts()
        macro(f'eone{tag}Harm', str(vv.get('Harm', 0)))
        macro(f'eone{tag}None', str(vv.get('No detectable effect', 0)))
        macro(f'eone{tag}Benefit', str(vv.get('Benefit', 0)))
        macro(f'eone{tag}Cells', str(len(s[s.dataset == ds])))
    frp = s[(s.dataset == 'frp_confined_concrete') & (s.n >= 100) & s.gen.isin(['cart', 'ctgan', 'copulagan', 'tvae'])]
    macro('frpBigHarm', str((frp.verdict == 'Harm').sum()))
    macro('frpBigCells', str(len(frp)))
    macro('frpBigMin', f'{100 * frp.median_d.min():.0f}')
    macro('frpBigMax', f'{100 * frp.median_d.max():.0f}')
    gc = reg[reg.generator == 'GaussianCopula']
    macro('gcHarm', str((gc.verdict == 'Harm').sum()))
    macro('gcCells', str(len(gc)))
    macro('gcMin', f'{100 * gc.median_d.min():.0f}')
    macro('gcMax', f'{100 * gc.median_d.max():.0f}')
    macro('eoneMaxHarm', f'{100 * reg.median_d.max():.0f}')
    fails = sum(int(pd.read_csv(os.path.join(e1dir, f)).failed.sum()) for f in
                ['e1_frp.csv', 'e1_cfst_rect.csv', 'e1x_cfst_circ_exploratory.csv'])
    macro('eoneFailures', str(fails))
    # real-only learning curves (medians over draws) for context
    for f, tag in [('e1_frp.csv', 'Frp'), ('e1_cfst_rect.csv', 'Rect')]:
        d = pd.read_csv(os.path.join(e1dir, f))
        r = d[d.generator == 'RealOnly'].groupby(['n', 'regressor']).r2.median()
        macro(f'rtwo{tag}Small', f'{r.loc[d.n.min()].min():.2f}')
        macro(f'rtwo{tag}SmallHi', f'{r.loc[d.n.min()].max():.2f}')
        macro(f'rtwo{tag}Large', f'{r.loc[d.n.max()].min():.2f}')
        macro(f'rtwo{tag}LargeHi', f'{r.loc[d.n.max()].max():.2f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--e0', required=True)
    ap.add_argument('--e0-compare', required=True)
    ap.add_argument('--e1', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fidelity_table(a.e0, a.out)
    regression_table(a.e0, a.e0_compare, a.out)
    e1_tables(a.e1, a.out)
    with open(os.path.join(a.out, 'numbers.tex'), 'w') as fh:
        fh.write(f'% generated by scripts/paper_tables.py from {a.e0}, {a.e0_compare}, {a.e1}\n')
        for k, v in macros.items():
            fh.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
    print(f'wrote {len(macros)} macros and 4 tables to {a.out}')
    for k, v in macros.items():
        print(f'  {k} = {v}')


if __name__ == '__main__':
    main()
