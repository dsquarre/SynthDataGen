import pandas as pd
import numpy as np
df = pd.read_excel('dataset.xlsx')
df.drop(columns=['Specimen'],inplace=True)
df.reset_index(drop=True,inplace=True)
df.columns = df.columns.str.strip()
from py_synthpop.synthpop.processor.missing_data_handler import MissingDataHandler
md_handler = MissingDataHandler()
#Get data types
metadata= md_handler.get_column_dtypes(df)
missingness_dict = md_handler.detect_missingness(df)
real_df = md_handler.apply_imputation(df, missingness_dict)

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
'''#CopulaGANSynthesizer
from sdv.single_table import CopulaGANSynthesizer
synthesizer = CopulaGANSynthesizer(metadata)
synthesizer.fit(real_df)
synthetic_data = synthesizer.sample(num_rows=100)
print(synthetic_data.head())
'''
'''#TVAESynthesizer
from sdv.single_table import TVAESynthesizer
synthesizer = TVAESynthesizer(metadata)
synthesizer.fit(real_df)
synthetic_data = synthesizer.sample(num_rows=100)
print(synthetic_data.head())
'''
