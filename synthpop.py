import pandas as pd
import numpy as np

df = pd.read_excel('dataset.xlsx')
df.drop(columns=['Specimen'],inplace=True)
df.reset_index(drop=True,inplace=True)
df.columns = df.columns.str.strip()
#df['Specimen']
#print(df.isnull().sum())

from py_synthpop.synthpop.processor.missing_data_handler import MissingDataHandler
from py_synthpop.synthpop.processor.data_processor import DataProcessor
from py_synthpop.synthpop.method.cart import CARTMethod
md_handler = MissingDataHandler()
# 1.1 Get data types
metadata= md_handler.get_column_dtypes(df)

missingness_dict = md_handler.detect_missingness(df)
real_df = md_handler.apply_imputation(df, missingness_dict)

processor = DataProcessor(metadata)
# 3.1 Preprocess the data: transforms raw data into a numerical format
processed_data = processor.preprocess(real_df)

cart = CARTMethod(metadata,minibucket=5, random_state=59)
cart.fit(processed_data)
synthetic_processed = cart.sample(100)
synthetic_df = processor.postprocess(synthetic_processed)
print(synthetic_df)