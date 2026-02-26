from synthpop import MetricsReport as SPMetricsReport, EfficacyMetrics
from metadata_manager import generate_metadata
import pandas as pd
from typing import Dict, Any
import os


def compute_report_for_pair(real_df: pd.DataFrame, synth_df: pd.DataFrame, metadata: Dict[str, Any]):
    """Compute synthpop MetricsReport and return its result as DataFrame."""
    report = SPMetricsReport(real_df, synth_df, metadata)
    report_result = report.generate_report()
    if not isinstance(report_result, pd.DataFrame):
        report_result = pd.DataFrame(report_result)
    return report_result


def evaluate_all(real_df: pd.DataFrame, outputs: Dict[str, Any], metadata: Dict[str, Any] = None, output_prefix: str = ''):
    """Evaluate and save all reports (real and synthetic) into a single Excel file for easy comparison.

    - `outputs` maps model_name -> DataFrame or file path.
    - Returns a dict model_name -> {'report': report_df, 'regression_metrics': metrics}
    """
    if metadata is None:
        metadata = generate_metadata(real_df)

    results = {}
    target_column = real_df.columns[-1]
    reg_efficacy = EfficacyMetrics(task='regression', target_column=target_column)

    combined_report_path = f"{output_prefix}combined_report.xlsx"
    excel_sheets = {}
    regression_metrics_list = []

    # Regression metrics for real data (real vs real)
    real_reg_metrics = reg_efficacy.evaluate(real_df, real_df)
    real_report_df = compute_report_for_pair(real_df, real_df, metadata)
    excel_sheets['real_report'] = real_report_df
    results['real'] = {'regression_metrics': real_reg_metrics}
    
    # Add real metrics to summary
    real_metrics_dict = {'Model': 'Real'}
    if isinstance(real_reg_metrics, dict):
        real_metrics_dict.update(real_reg_metrics)
    regression_metrics_list.append(real_metrics_dict)

    for name, val in outputs.items():
        if isinstance(val, str):
            # try csv then excel
            try:
                synth_df = pd.read_csv(val)
            except Exception:
                synth_df = pd.read_excel(val)
        else:
            synth_df = val

        report_df = compute_report_for_pair(real_df, synth_df, metadata)
        reg_metrics = reg_efficacy.evaluate(real_df, synth_df)

        sheet_base = name.replace(' ', '_')
        excel_sheets[f'{sheet_base}_report'] = report_df

        results[name] = {'report': report_df, 'regression_metrics': reg_metrics}
        
        # Add to summary
        metrics_dict = {'Model': name}
        if isinstance(reg_metrics, dict):
            metrics_dict.update(reg_metrics)
        regression_metrics_list.append(metrics_dict)

    # Create summary sheet with all regression metrics side-by-side
    summary_df = pd.DataFrame(regression_metrics_list)
    excel_sheets['Regression_Summary'] = summary_df

    # Write all sheets to a single Excel file
    with pd.ExcelWriter(combined_report_path) as writer:
        for sheet, df in excel_sheets.items():
            safe_sheet = sheet[:31]
            df.to_excel(writer, sheet_name=safe_sheet, index=False)

    results['combined_report_path'] = combined_report_path
    return results


if __name__ == '__main__':
    real_df = pd.read_csv('data.csv')
    try:
        real_df.drop(columns=real_df.columns[0], inplace=True)
    except Exception:
        pass
    real_df.columns = real_df.columns.str.strip()
    metadata = generate_metadata(real_df)

    # collect expected synthetic outputs if present
    outputs = {}
    candidates = ['cart_data.xlsx', 'gc_data.xlsx', 'gc_synthetic_data.xlsx', 'ctgan_synthetic_data.xlsx', 'tvae_synthetic_data.xlsx', 'copula_gan_synthetic_data.xlsx']
    for name in candidates:
        if os.path.exists(name):
            outputs[name.replace('.xlsx', '')] = name

    if outputs:
        res = evaluate_all(real_df, outputs, metadata)
        print(f"Report written to: {res.get('combined_report_path')}")
