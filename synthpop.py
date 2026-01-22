import pandas as pd
import numpy as np

df = pd.read_excel('dataset.xlsx')
df.drop(columns=['Specimen'],inplace=True)
df.reset_index(drop=True,inplace=True)
df.columns = df.columns.str.strip()
#print(df.columns)

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
cart_df = processor.postprocess(synthetic_processed)
cart_df.to_excel('cart_df.xlsx') 

gc = GaussianCopulaMethod(metadata)
gc.fit(processed_data)
synth_gc = gc.sample(100)
gc_df = processor.postprocess(synth_gc)
gc_df.to_excel('gc_df.xlsx') 

from py_synthpop.synthpop.metrics.diagnostic_report import MetricsReport
report_cart = MetricsReport(real_df, cart_df, metadata)
report_df_cart = report_cart.generate_report()
#print(report_df)
report_df_cart.to_excel('cart_report.xlsx')

report_gc = MetricsReport(real_df, gc_df, metadata)
report_df_gc = report_gc.generate_report()
report_df_gc.to_excel('gc_report.xlsx')
from py_synthpop.synthpop.metrics.efficacy_metrics import EfficacyMetrics
reg_efficacy = EfficacyMetrics(task='regression', target_column="Pu [kN]")
print('Regression Efficacy Metrics for CART:')
reg_metrics_cart = reg_efficacy.evaluate(real_df, cart_df)
print(reg_metrics_cart,'\n')

print('Regression Efficacy Metrics for GC:')
reg_metrics_gc = reg_efficacy.evaluate(real_df, gc_df)
print(reg_metrics_gc)