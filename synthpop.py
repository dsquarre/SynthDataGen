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
from py_synthpop.synthpop.method.GC import GaussianCopulaMethod
md_handler = MissingDataHandler()
#Get data types
metadata= md_handler.get_column_dtypes(df)

missingness_dict = md_handler.detect_missingness(df)
real_df = md_handler.apply_imputation(df, missingness_dict)

processor = DataProcessor(metadata)
#Preprocess the data: transforms raw data into a numerical format
processed_data = processor.preprocess(real_df)

cart = CARTMethod(metadata,minibucket=5, random_state=59)
cart.fit(processed_data)
synthetic_processed = cart.sample(100)
synthetic_df = processor.postprocess(synthetic_processed)
print(synthetic_df)

gc = GaussianCopulaMethod(metadata)
gc.fit(processed_data)
synth_gc = gc.sample(100)
gc_df = processor.postprocess(synth_gc)
print(gc_df)


from py_synthpop.synthpop.metrics.diagnostic_report import MetricsReport
report = MetricsReport(real_df, synthetic_df, metadata)
report_df = report.generate_report()
print(report_df)

from py_synthpop.synthpop.metrics.efficacy_metrics import EfficacyMetrics
reg_efficacy = EfficacyMetrics(task='regression', target_column="income")
reg_metrics = reg_efficacy.evaluate(real_df, synthetic_df)
