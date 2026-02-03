import pandas as pd
import numpy as np
real_df = pd.read_excel('imputed_dataset.xlsx')

from sdv.metadata import Metadata
metadata = Metadata.load_from_json('sdv_metadata.json')
print(real_df.columns)
'''
##single table GC
from sdv.single_table import GaussianCopulaSynthesizer
synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(real_df)
synthetic_data = synthesizer.sample(num_rows=500)
#print(synthetic_data.head())
'''
'''#CTGANSynthesizer
from sdv.single_table import CTGANSynthesizer
synthesizer = CTGANSynthesizer(metadata)
synthesizer.fit(real_df)
synthetic_data = synthesizer.sample(num_rows=500)
#print(synthetic_data.head())
'''
#CopulaGANSynthesizer
from sdv.single_table import CopulaGANSynthesizer
synthesizer = CopulaGANSynthesizer(metadata)
synthesizer.fit(real_df)
synthetic_data = synthesizer.sample(num_rows=100)
print(synthetic_data.head())

'''#TVAESynthesizer
from sdv.single_table import TVAESynthesizer
synthesizer = TVAESynthesizer(metadata)
synthesizer.fit(real_df)
synthetic_data = synthesizer.sample(num_rows=100)
print(synthetic_data.head())
'''
