"""Parse Tables 3-7 (FRP-confined concrete test database) from Ozbakkaloglu & Lim (2013) accepted manuscript.

Input: pdftotext -layout output. Output: CSV with one row per specimen.
Column assignment: per page, numeric-token centres are clustered into columns and matched to the
15 header columns with a per-page linear map x_page = a * x_header + b (layout mode rescales each page).
"""
import itertools
import re
import sys

import numpy as np
import pandas as pd

COLS = ['D_mm', 'H_mm', 'fco_MPa', 'eco_pct', 'Efrp_GPa', 'ffrp_MPa', 'tfrp_mm', 'Ef_GPa', 'ff_MPa', 'tf_mm',
        'fcc_MPa', 'ecu_pct', 'eh_rup_pct', 'k_eps_frp', 'k_eps_f']
HEADER_LABELS = ['D', 'H', "f'co", 'εco', 'Efrp', 'ffrp', 'tfrp', 'Ef', 'ff', 'tf', "f'cc", 'εcu', 'εh,rup']
TABLES = {3: 'CFRP', 4: 'GFRP', 5: 'AFRP', 6: 'HM/UHM CFRP', 7: 'Unbonded wrap or tube'}

text = open(sys.argv[1], encoding='utf-8').read()
pages = text.split('\f')
row_re = re.compile(r'^\s*([A-Z][^\[]*?\[\d+\])\s')
num_re = re.compile(r'^(\d+\.?\d*|\.\d+)([a-zA-Z^*#†,]*)$')


def header_centres(line):
    """Centres of the 13 labelled columns + 2 k-factor columns extrapolated from spacing."""
    cents = []
    pos = 0
    for lab in HEADER_LABELS:
        i = line.index(lab, pos)
        cents.append(i + len(lab) / 2)
        pos = i + len(lab)
    step = cents[-1] - cents[-2]
    cents += [cents[-1] + 1.4 * step, cents[-1] + 2.4 * step]  # kε,frp and kε,f sit on the next header line
    return np.array(cents)


def cluster(xs, gap=2.5):
    xs = sorted(xs)
    groups = [[xs[0]]]
    for x in xs[1:]:
        (groups[-1].append(x) if x - groups[-1][-1] <= gap else groups.append([x]))
    return [np.mean(g) for g in groups], [len(g) for g in groups]


def fit_map(cl_centres, hdr):
    """Find a,b and a monotone assignment of clusters to header columns minimising squared error."""
    best = None
    k = len(cl_centres)
    for i, j in itertools.combinations(range(len(hdr)), 2):
        if j - i < k - 1:
            continue
        a = (cl_centres[-1] - cl_centres[0]) / (hdr[j] - hdr[i])
        b = cl_centres[0] - a * hdr[i]
        mapped = a * hdr + b
        assign = [int(np.argmin(abs(mapped - c))) for c in cl_centres]
        if len(set(assign)) != k or assign != sorted(assign):
            continue
        err = sum((mapped[t] - c) ** 2 for t, c in zip(assign, cl_centres))
        if best is None or err < best[0]:
            best = (err, a, b, assign)
    return best


def assign_monotone(xs, mapped):
    """Assign ordered token centres xs to distinct, increasing columns minimising total |x - column centre|."""
    n, m = len(xs), len(mapped)
    INF = float('inf')
    cost = [[INF] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    for j in range(m + 1):
        cost[0][j] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            skip = cost[i][j - 1]                                   # column j-1 left empty
            take = cost[i - 1][j - 1] + abs(xs[i - 1] - mapped[j - 1])  # token i-1 -> column j-1
            if take <= skip:
                cost[i][j], back[i][j] = take, 'take'
            else:
                cost[i][j], back[i][j] = skip, 'skip'
    out, i, j = [None] * n, n, m
    while i > 0:
        if back[i][j] == 'take':
            out[i - 1] = j - 1
            i, j = i - 1, j - 1
        else:
            j -= 1
    return out


records = []

def build_records(pi, current_table, rows, mapped):
    for paper, frp_type, nums, raw in rows:
        rec = {'raw_line': raw, 'Table': current_table, 'Confinement': TABLES[current_table], 'FRP_type_detail': frp_type,
               'Source': paper, 'Page': pi + 1}
        cols_for = assign_monotone([(s + e) / 2 for s, e, _ in nums], list(mapped))
        for (s, e, tok), col in zip(nums, cols_for):
            val, flag = num_re.match(tok).groups()
            name = COLS[col]
            if name in rec:
                rec.setdefault('_collisions', []).append(name); print('COLLIDE p', pi + 1, paper, name, rec[name], tok)
            rec[name] = float(val)
            if flag:
                rec[name + '_note'] = flag
        records.append(rec)



current_table = None
hdr = None
diag = []
templates = {}
ref_vals = [[] for _ in COLS]
pending = []
for pi, pg in enumerate(pages):
    if re.search(r'Table [12]\. Summary', pg) or re.search(r'Table (8|9|10|11|12)\. (Average|Variation|Statistics)', pg):
        current_table = None
        continue
    m = re.search(r'Table (\d)\. Test database of', pg)
    hdr_lines = [l for l in pg.split('\n') if "f'co" in l and 'Efrp' in l and 'εh,rup' in l]
    if m and int(m.group(1)) in TABLES and hdr_lines:
        current_table = int(m.group(1))
        hdr = header_centres(hdr_lines[0])
    if current_table is None or hdr is None:
        continue
    rows = []
    for line in pg.split('\n'):
        mm = row_re.match(line)
        if not mm:
            continue
        toks = [(t.start(), t.end(), t.group()) for t in re.finditer(r'\S+', line) if t.start() >= mm.end(1)]
        frp_type = ''
        while toks and not num_re.match(toks[0][2]):
            frp_type += toks.pop(0)[2]
        nums = [t for t in toks if num_re.match(t[2])]
        if len(nums) < 4:
            continue
        rows.append((mm.group(1).strip(), frp_type, nums, line.strip()))
    if not rows:
        continue
    cents, counts = cluster([(s + e) / 2 for _, _, nums, _ in rows for s, e, _ in nums])
    cents = np.array(cents)
    vals_by_cluster = [[] for _ in cents]
    for _, _, nums, _ in rows:
        for st_, en_, tok in nums:
            ci = int(np.argmin(abs(cents - (st_ + en_) / 2)))
            vals_by_cluster[ci].append(float(num_re.match(tok).group(1)))
    if len(cents) == len(COLS):
        mapped = cents                      # every column present: clusters are the columns, in order
        for ci, v in enumerate(vals_by_cluster):
            ref_vals[ci].extend(v)
        err = 0.0
    else:
        pending.append((pi, current_table, rows, cents, counts, vals_by_cluster))
        continue
    diag.append((pi + 1, current_table, len(rows), len(cents), round(err, 1)))
    build_records(pi, current_table, rows, mapped)

ref_med = np.array([np.median(np.log(np.clip(v, 1e-3, None))) for v in ref_vals])
ref_sd = np.array([max(np.std(np.log(np.clip(v, 1e-3, None))), 0.15) for v in ref_vals])
for pi, current_table, rows, cents, counts, vals_by_cluster in pending:
    keep_cl = [i for i, c in enumerate(counts) if c > 3] if len(cents) > len(COLS) else list(range(len(cents)))
    best = None
    for cols in itertools.combinations(range(len(COLS)), len(keep_cl)):
        if not {0, 1, 2, 10} <= set(cols):
            continue
        score = sum(abs(np.median(np.log(np.clip(vals_by_cluster[ci], 1e-3, None))) - ref_med[c]) / ref_sd[c]
                    for ci, c in zip(keep_cl, cols))
        if best is None or score < best[0]:
            best = (score, cols)
    # column centres for this page: known clusters at their assigned columns, gaps interpolated
    known = dict(zip(best[1], cents[keep_cl]))
    xs = np.array(sorted(known)); ys = np.array([known[c] for c in xs])
    mapped = np.interp(np.arange(len(COLS)), xs, ys, left=None, right=None)
    for c in range(len(COLS)):
        if c > xs.max():
            mapped[c] = ys[-1] + (c - xs.max()) * np.median(np.diff(ys) / np.diff(xs))
        if c < xs.min():
            mapped[c] = ys[0] - (xs.min() - c) * np.median(np.diff(ys) / np.diff(xs))
    print('PARTIAL page', pi + 1, 'clusters->cols', dict(zip([round(cents[i]) for i in keep_cl], [COLS[c] for c in best[1]])))
    diag.append((pi + 1, current_table, len(rows), len(cents), round(best[0], 1)))
    build_records(pi, current_table, rows, mapped)

df = pd.DataFrame(records)
# Identity check k = eh_rup / rupture strain, rupture strain(%) = f / (10 E). If eh_rup is empty but both k
# columns are filled and the values read as (eh_rup, k_eps_frp) satisfy the FRP identity, shift them.
df['parse_fix'] = ''
for i, r in df.iterrows():
    if pd.isna(r.get('eh_rup_pct')) and pd.notna(r.get('k_eps_frp')) and pd.notna(r.get('k_eps_f')) \
            and pd.notna(r.get('ffrp_MPa')) and pd.notna(r.get('Efrp_GPa')):
        rup = r['ffrp_MPa'] / (10 * r['Efrp_GPa'])
        if abs(r['k_eps_frp'] / rup - r['k_eps_f']) / r['k_eps_f'] < 0.05:
            df.loc[i, ['eh_rup_pct', 'k_eps_frp', 'k_eps_f']] = [r['k_eps_frp'], r['k_eps_f'], np.nan]
            if 'k_eps_f_note' in df and pd.notna(r.get('k_eps_f_note')):
                df.loc[i, 'k_eps_frp_note'] = r['k_eps_f_note']; df.loc[i, 'k_eps_f_note'] = np.nan
            df.loc[i, 'parse_fix'] = 'shifted eh_rup/k by identity'
            print('FIXED', r['Source'], 'page', r['Page'])

for (p, t, n, k, e) in diag:
    print(f'page {p:2d} table {t}: rows {n:3d}, clusters {k:2d}, fit error {e}')
print('rows per table:', df.Table.value_counts().sort_index().to_dict(), 'total', len(df))
if '_collisions' in df:
    print('rows with two values in one column:', df['_collisions'].notna().sum())
df.drop(columns=[c for c in ['_collisions'] if c in df], inplace=True)
ordered = ['Table', 'Confinement', 'FRP_type_detail', 'Source'] + COLS + sorted(c for c in df.columns if c.endswith('_note')) + ['parse_fix', 'Page', 'raw_line']
df[[c for c in ordered if c in df.columns]].to_csv(sys.argv[2], index=False)
print('wrote', sys.argv[2])
