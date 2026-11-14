import json
import logging
import warnings

from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, CopulaGANSynthesizer, TVAESynthesizer

from seeding import next_seed, seed_everything

warnings.filterwarnings('ignore')
for _logger in ('sdv', 'rdt', 'copulas', 'SingleTableSynthesizer', 'GaussianCopulaSynthesizer',
                'CTGANSynthesizer', 'CopulaGANSynthesizer', 'TVAESynthesizer'):
    logging.getLogger(_logger).setLevel(logging.ERROR)


def _fit_and_sample(synthesizer_cls, df, num_rows):
    """Fit an SDV synthesizer on df and sample num_rows, seeded per call."""
    with open('sdv_metadata.json', 'r', encoding='utf-8') as f:
        metadata = Metadata.load_from_dict(json.load(f))
    seed = next_seed()
    seed_everything(seed)  # controls fitting (torch / numpy)
    synthesizer = synthesizer_cls(metadata)
    synthesizer.fit(df)
    # SDV 1.x samples with a fixed internal seed unless one is set, which made
    # every batch identical; pass the per-call seed instead (private API, SDV<2).
    synthesizer._set_random_state(seed)
    return synthesizer.sample(num_rows=num_rows)


# output_prefix is unused but kept so all generators share one call signature.
def run_gaussian_copula(df, num_rows, output_prefix):
    return 'gaussian_copula', _fit_and_sample(GaussianCopulaSynthesizer, df, num_rows)


def run_ctgan(df, num_rows, output_prefix):
    return 'ctgan', _fit_and_sample(CTGANSynthesizer, df, num_rows)


def run_copula_gan(df, num_rows, output_prefix):
    return 'copulagan', _fit_and_sample(CopulaGANSynthesizer, df, num_rows)


def run_tvae(df, num_rows, output_prefix):
    return 'tvae', _fit_and_sample(TVAESynthesizer, df, num_rows)
