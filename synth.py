import pandas as pd
import numpy as np
from synthpop import DataProcessor, CARTMethod, GaussianCopulaMethod
def run_cart(df, cart_samples,impute,metadata):
	processor = DataProcessor(metadata)
	processed_data = processor.preprocess(df)
	cart = CARTMethod(metadata, minibucket=5, random_state=59)
	cart.fit(processed_data)
	synthetic_processed = cart.sample(cart_samples)
	cart_df = processor.postprocess(synthetic_processed)
	cart_df.to_excel(f'datasets/{impute}cart_data.xlsx', index=False)
	return 'cart',cart_df

def synth_gc(df,gc_samples,impute,metadata):
	processor = DataProcessor(metadata)
	processed_data = processor.preprocess(df)
	gc = GaussianCopulaMethod(metadata)
	gc.fit(processed_data)
	synth_gc = gc.sample(gc_samples)
	gc_df = processor.postprocess(synth_gc)
	gc_df.to_excel(f'datasets/gc_data.xlsx', index=False)
	return gc_df
	
