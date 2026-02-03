import pandas as pd
import numpy as np


df = pd.read_excel('dataset.xlsx')
df.drop(columns=['Specimen'],inplace=True)
df.reset_index(drop=True,inplace=True)
df.columns = df.columns.str.strip()

from synthpop import MissingDataHandler, DataProcessor, CARTMethod, GaussianCopulaMethod, MetricsReport, EfficacyMetrics  

md_handler = MissingDataHandler()
metadata= md_handler.get_column_dtypes(df)

missingness_dict = md_handler.detect_missingness(df)
real_df = md_handler.apply_imputation(df, missingness_dict)
real_df.to_excel('imputed_dataset.xlsx', index=False)
processor = DataProcessor(metadata)
#Preprocess the data: transforms raw data into a numerical format
processed_data = processor.preprocess(real_df)

cart = CARTMethod(metadata,minibucket=5, random_state=59)
cart.fit(processed_data)
synthetic_processed = cart.sample(100)
cart_df = processor.postprocess(synthetic_processed)
print(cart_df.head())

gc = GaussianCopulaMethod(metadata)
gc.fit(processed_data)
synth_gc = gc.sample(100)
gc_df = processor.postprocess(synth_gc)
print(gc_df.head() )

report_cart = MetricsReport(real_df, cart_df, metadata)
report_df_cart = report_cart.generate_report()
#print(report_df)
print(report_df_cart)
report_gc = MetricsReport(real_df, gc_df, metadata)
report_df_gc = report_gc.generate_report()
print(report_df_gc)
#report_df_gc.to_excel('gc_report.xlsx')


reg_efficacy = EfficacyMetrics(task='regression', target_column="Pu [kN]")
print('Regression Efficacy Metrics for CART:')
reg_metrics_cart = reg_efficacy.evaluate(real_df, cart_df)
print(reg_metrics_cart,'\n')

print('Regression Efficacy Metrics for GC:')
reg_metrics_gc = reg_efficacy.evaluate(real_df, gc_df)
print(reg_metrics_gc)