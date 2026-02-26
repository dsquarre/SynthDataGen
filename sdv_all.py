import pandas as pd
import numpy as np
import json
from sdv.metadata import Metadata


def run_sdv_models(df: pd.DataFrame = None, sdv_metadata_path: str = 'sdv_metadata.json', num_rows: int = 500, output_prefix: str = ''):
    """Run SDV single-table synthesizers using Metadata loaded from a JSON file.

    Returns a dict of generated DataFrames keyed by synthesizer name.
    """
    if df is None:
        df = pd.read_csv('data.csv')
        df.drop(columns=df.columns[0], inplace=True)
        df.columns = df.columns.str.strip()

    # Load SDV metadata from JSON
    with open(sdv_metadata_path, 'r', encoding='utf-8') as f:
        meta_dict = json.load(f)
    print(meta_dict)
    metadata = Metadata.load_from_dict(meta_dict)

    outputs = {}

    # Gaussian Copula
    from sdv.single_table import GaussianCopulaSynthesizer
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(df)
    synthetic_data = synthesizer.sample(num_rows=num_rows)
    synthetic_data.to_excel(f'{output_prefix}gc_synthetic_data.xlsx', index=False)
    outputs['gaussian_copula'] = synthetic_data

    # CTGAN
    from sdv.single_table import CTGANSynthesizer
    synthesizer = CTGANSynthesizer(metadata)
    synthesizer.fit(df)
    synthetic_data = synthesizer.sample(num_rows=num_rows)
    synthetic_data.to_excel(f'{output_prefix}ctgan_synthetic_data.xlsx', index=False)
    outputs['ctgan'] = synthetic_data

    # CopulaGAN
    from sdv.single_table import CopulaGANSynthesizer
    synthesizer = CopulaGANSynthesizer(metadata)
    synthesizer.fit(df)
    synthetic_data = synthesizer.sample(num_rows=num_rows)
    synthetic_data.to_excel(f'{output_prefix}copula_gan_synthetic_data.xlsx', index=False)
    outputs['copulagan'] = synthetic_data

    # TVAE
    from sdv.single_table import TVAESynthesizer
    synthesizer = TVAESynthesizer(metadata)
    synthesizer.fit(df)
    synthetic_data = synthesizer.sample(num_rows=num_rows)
    synthetic_data.to_excel(f'{output_prefix}tvae_synthetic_data.xlsx', index=False)
    outputs['tvae'] = synthetic_data

    return outputs
