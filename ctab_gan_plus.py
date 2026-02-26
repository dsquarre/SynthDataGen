import os
import tempfile
from pathlib import Path
import pandas as pd


def _find_output_csv(output_dir: str):
    p = Path(output_dir)
    csvs = list(p.glob('**/*.csv'))
    if not csvs:
        return None
    return str(max(csvs, key=lambda p: p.stat().st_size))


def run_ctab_gan_plus(df: pd.DataFrame = None, data_csv: str = None, num_samples: int = 500, output_dir: str = None):
    """Invoke installed CTAB-GAN-Plus package to train and sample synthetic data.

    This wrapper assumes a CTAB-GAN-Plus Python package is already installed. The package API can vary
    between versions; this function attempts common entrypoints and otherwise raises a helpful error
    instructing the user how to run the package manually.

    Returns path to produced CSV when successful.
    """
    if data_csv is None and df is None:
        raise ValueError("Either df or data_csv must be provided")
    if data_csv is None:
        tmpf = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        df.to_csv(tmpf.name, index=False)
        data_csv = tmpf.name

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix='ctab_gan_plus_out_')
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Try importing likely package names
    candidates = ['ctab_gan_plus', 'ctabgan_plus', 'ctabgan', 'ctab_gan']
    mod = None
    for name in candidates:
        try:
            mod = __import__(name)
            break
        except Exception:
            mod = None
    if mod is None:
        raise RuntimeError("CTAB-GAN-Plus package not found. Please ensure it is installed (e.g. pip install CTAB-GAN-Plus) and available in Python path.")

    # Attempt common API patterns
    if hasattr(mod, 'train'):
        # train(data_path, output_dir, samples)
        try:
            mod.train(data_csv, output_dir=output_dir, samples=num_samples)
        except TypeError:
            mod.train(data_csv, output_dir)
    elif hasattr(mod, 'run'):
        try:
            mod.run(data_csv, output_dir=output_dir, samples=num_samples)
        except TypeError:
            mod.run(data_csv, output_dir)
    elif hasattr(mod, 'CTABGAN') or hasattr(mod, 'CTABGANPlus') or hasattr(mod, 'CTAB_GAN_Plus'):
        # Try to instantiate and call fit/sample
        cls = None
        for cname in ('CTABGANPlus', 'CTABGAN', 'CTAB_GAN_Plus'):
            cls = getattr(mod, cname, None)
            if cls:
                break
        if cls is None:
            raise RuntimeError('Installed CTAB-GAN-Plus package found but no known trainer class available.')
        try:
            trainer = cls()
            if hasattr(trainer, 'fit'):
                trainer.fit(data_csv)
            elif hasattr(trainer, 'train'):
                trainer.train(data_csv)
            # try sampling
            if hasattr(trainer, 'sample'):
                trainer.sample(num_samples, output_dir=output_dir)
        except Exception as e:
            raise RuntimeError(f'Failed to run CTAB-GAN-Plus API: {e}')
    else:
        raise RuntimeError('Installed CTAB-GAN-Plus package found but automatic invocation is not implemented for this package layout. Please consult package docs.')

    csv_path = _find_output_csv(output_dir)
    if csv_path:
        return csv_path
    raise RuntimeError(f'No CSV produced in {output_dir}. Check CTAB-GAN-Plus output directory for results.')


if __name__ == '__main__':
    print('This module provides run_ctab_gan_plus(df=..., data_csv=..., output_dir=...)')
