import pandas as pd
import numpy as np
import json
import logging
import warnings

warnings.filterwarnings('ignore')
logging.getLogger('sdv').setLevel(logging.ERROR)
logging.getLogger('rdt').setLevel(logging.ERROR)
logging.getLogger('copulas').setLevel(logging.ERROR)
logging.getLogger('SingleTableSynthesizer').setLevel(logging.ERROR)
logging.getLogger('GaussianCopulaSynthesizer').setLevel(logging.ERROR)
logging.getLogger('CTGANSynthesizer').setLevel(logging.ERROR)
logging.getLogger('CopulaGANSynthesizer').setLevel(logging.ERROR)
logging.getLogger('TVAESynthesizer').setLevel(logging.ERROR)

from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, CopulaGANSynthesizer, TVAESynthesizer

def run_gaussian_copula(df, num_rows, output_prefix):
    with open('sdv_metadata.json', 'r', encoding='utf-8') as f:
        meta_dict = json.load(f)
    metadata = Metadata.load_from_dict(meta_dict)
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(df)
    synthetic_data = synthesizer.sample(num_rows=num_rows)
    return 'gaussian_copula', synthetic_data

def run_ctgan(df, num_rows, output_prefix):
    with open('sdv_metadata.json', 'r', encoding='utf-8') as f:
        meta_dict = json.load(f)
    metadata = Metadata.load_from_dict(meta_dict)
    synthesizer = CTGANSynthesizer(metadata)
    synthesizer.fit(df)
    synthetic_data = synthesizer.sample(num_rows=num_rows)
    return 'ctgan',synthetic_data

def run_copula_gan(df, num_rows, output_prefix):
    with open('sdv_metadata.json', 'r', encoding='utf-8') as f:
        meta_dict = json.load(f)
    metadata = Metadata.load_from_dict(meta_dict)
    synthesizer = CopulaGANSynthesizer(metadata)
    synthesizer.fit(df)
    synthetic_data = synthesizer.sample(num_rows=num_rows)
    synthetic_data.to_excel(f'datasets/{output_prefix}copula_gan_synthetic_data.xlsx', index=False)
    return 'copulagan', synthetic_data

def run_tvae(df, num_rows, output_prefix):
    with open('sdv_metadata.json', 'r', encoding='utf-8') as f:
        meta_dict = json.load(f)
    metadata = Metadata.load_from_dict(meta_dict)
    synthesizer = TVAESynthesizer(metadata)
    synthesizer.fit(df)
    synthetic_data = synthesizer.sample(num_rows=num_rows)
    synthetic_data.to_excel(f'datasets/{output_prefix}tvae_synthetic_data.xlsx', index=False)
    return 'tvae', synthetic_data
