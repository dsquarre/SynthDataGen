"""One-time preparation of the RACFST cyclic database (Xu et al. 2020, Appendix A3) for run_all.py.

    python scripts/prepare_racfst.py [--src datasets/dataset.xlsx] [--out datasets/racfst_cyclic.xlsx]

Writes a pipeline-ready file with identifier-style column names. The matching run configuration is
configs/racfst.json, which names the target, the class column and the derived ratios by column name.

Decisions made here (documented so they are not repeated by hand in every run):
- Removed as inputs: displacement ductility mu and equivalent viscous damping xi. They are measured outcomes of the
  same cyclic tests as the target Pu, so a designer would not know them before testing.
- Removed and rebuilt by run_all.py from their components (configs/racfst.json "derived"): w/c = water / cement,
  D/t = D / t, and L/D = (2L / 2) / D, i.e. L is half the reported length 2L. The generators therefore never
  have to learn these identities.
- The source "L/D" column is inconsistent: for some studies it equals (2L/2)/D, for others 2L/D, and one specimen
  matches neither. The rebuilt L/D uses (2L/2)/D for every test.
- RFA (recycled fine aggregate) is removed: it is 0 kg/m3 for all 56 tests.
"""
import argparse

import pandas as pd

RENAME = {
    'Specimen': 'Specimen',
    'Section': 'Section',
    'D(or B)': 'D_mm',
    't': 't_mm',
    '2L [mm]': 'L2_mm',
    'r [%]': 'r_pct',
    'Cement [kg/m3]': 'cement_kgm3',
    'Sand [kg/m3]': 'sand_kgm3',
    'NCA [kg/m3]': 'nca_kgm3',
    'RCA [kg/m3]': 'rca_kgm3',
    'Water [kg/m3]': 'water_kgm3',
    'fcu [MPa]': 'fcu_MPa',
    'fy [MPa]': 'fy_MPa',
    'n': 'n',
    'Pu [kN]': 'Pu_kN',
}
DROPPED = {
    'w/c': 'derived (water / cement), rebuilt by run_all.py',
    'D/t': 'derived (D / t), rebuilt by run_all.py',
    'L/D': 'inconsistent definition in source; rebuilt by run_all.py as (2L/2)/D',
    'RFA [kg/m3]': 'constant 0 in all tests',
    'μ': 'test outcome (ductility), not a design input',
    'ξ': 'test outcome (damping), not a design input',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='datasets/dataset.xlsx')
    ap.add_argument('--out', default='datasets/racfst_cyclic.xlsx')
    a = ap.parse_args()
    df = pd.read_excel(a.src)
    df.columns = df.columns.str.replace('\xa0', ' ').str.strip()
    unexpected = set(df.columns) - set(RENAME) - set(DROPPED)
    missing = (set(RENAME) | set(DROPPED)) - set(df.columns)
    if unexpected or missing:
        raise SystemExit(f'column mismatch: unexpected {sorted(unexpected)}, missing {sorted(missing)}')
    assert (df['RFA [kg/m3]'] == 0).all(), 'RFA is no longer constant; revisit the decision to drop it'
    out = df[list(RENAME)].rename(columns=RENAME)
    out.to_excel(a.out, index=False)
    print(f'wrote {a.out}: {out.shape[0]} rows, {out.shape[1]} columns')
    for col, why in DROPPED.items():
        print(f'  dropped {col!r}: {why}')


if __name__ == '__main__':
    main()
