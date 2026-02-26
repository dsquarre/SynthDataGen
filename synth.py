import pandas as pd
from synthpop import MissingDataHandler, DataProcessor, CARTMethod, GaussianCopulaMethod
from metadata_manager import generate_metadata


def run_synthpop(df: pd.DataFrame = None, metadata: dict = None, cart_samples: int = 500, gc_samples: int = 100, output_prefix: str = ''):
	"""Run synthpop CART and GaussianCopula methods and write outputs to Excel.

	Returns a dict of generated DataFrames: {'cart': cart_df, 'gc': gc_df}
	"""
	if df is None:
		df = pd.read_csv('data.csv')
		df.columns = df.columns.str.strip()

	md_handler = MissingDataHandler()
	if metadata is None:
		metadata = generate_metadata(df)

	missingness_dict = md_handler.detect_missingness(df)
	real_df = md_handler.apply_imputation(df, missingness_dict)

	processor = DataProcessor(metadata)
	processed_data = processor.preprocess(real_df)

	outputs = {}

	cart = CARTMethod(metadata, minibucket=5, random_state=59)
	cart.fit(processed_data)
	synthetic_processed = cart.sample(cart_samples)
	cart_df = processor.postprocess(synthetic_processed)
	cart_df.to_excel(f'{output_prefix}cart_data.xlsx', index=False)
	outputs['cart'] = cart_df

	gc = GaussianCopulaMethod(metadata)
	gc.fit(processed_data)
	synth_gc = gc.sample(gc_samples)
	gc_df = processor.postprocess(synth_gc)
	gc_df.to_excel(f'{output_prefix}gc_data.xlsx', index=False)
	outputs['gc'] = gc_df

	return outputs


# ...existing code...