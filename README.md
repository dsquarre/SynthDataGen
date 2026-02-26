# SynthDataGen

A comprehensive synthetic data generation toolkit that combines multiple state-of-the-art methods for creating realistic synthetic datasets. This project integrates CART, Gaussian Copula, SDV models, and CTAB-GAN+ with support for custom constraints and data augmentation.

## Features

- **Multiple Synthesis Methods:**
  - CART (Classification and Regression Trees)
  - Gaussian Copula (Synthpop)
  - SDV Gaussian Copula Synthesizer
  - CTGAN (Conditional Tabular GAN)
  - CopulaGAN
  - TVAE (Tabular Variational Autoencoder)
  - CTAB-GAN+ (deep learning approach)

- **Constraint Management:**
  - Interactive constraint definition
  - Custom logical constraints (e.g., formulas like `Volume = L*B*H`)
  - Automatic validation against numerical ranges
  - Constraint-based row filtering

- **Data Augmentation:**
  - Random Oversampling (ROS) with constraint validation
  - Iterative regeneration to meet target row counts
  - Metadata management and editing

- **Evaluation Metrics:**
  - Quality assessment of synthetic data
  - Comparison across different generation methods

## Project Structure

- `run_all.py` - Main orchestration script running all synthesis methods
- `synth.py` - Synthpop CART and Gaussian Copula implementations
- `sdv_all.py` - SDV single-table model implementations
- `ctab_gan_plus.py` - CTAB-GAN+ wrapper and integration
- `ros_augmentation.py` - Random Oversampling with constraint validation
- `metadata_manager.py` - Metadata generation and management
- `custom_constraints.py` - Custom constraint definition and validation
- `metrics.py` - Synthetic data quality evaluation
- `data.csv` - Input dataset (user-provided)
- `sdv_metadata.json` - SDV-compatible metadata configuration

## Installation

1. Clone the repository
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

**Note:** The python-synthpop package has some dependency issues, so the repository includes a manually modified version.

## Usage

### Quick Start

1. Place your dataset in `data.csv`
2. Run the complete pipeline:
   ```bash
   python3 run_all.py
   ```

### Interactive Workflow

The main script `run_all.py` guides you through:

1. **Data Loading:** Loads and prepares your CSV file
2. **Metadata Generation:** Creates column-type metadata
3. **Metadata Editing:** Interactively adjust column types (numeric 'n' or categorical 'c')
4. **Constraint Definition:** (Optional) Add custom constraints
5. **Synthesis:** Runs all available generation methods
6. **Evaluation:** Computes quality metrics across methods

### Custom Constraints

Add domain-specific constraints during the interactive workflow:
```
Example: w/c = Water/Cement, Volume = L*B*H
```

Constraints are saved to `constraints.json` for reproducibility.

## Output Files

- `cart_data.xlsx` - CART synthetic samples
- `gc_data.xlsx` - Synthpop Gaussian Copula synthetic samples
- `gc_synthetic_data.xlsx` - SDV Gaussian Copula Synthesizer output
- `ctgan_synthetic_data.xlsx` - CTGAN synthetic samples
- `copula_gan_synthetic_data.xlsx` - CopulaGAN synthetic samples
- `tvae_synthetic_data.xlsx` - TVAE synthetic samples
- `ctab_gan_plus_synthetic.csv` - CTAB-GAN+ output
- `metrics_report.csv` - Quality evaluation results
- `sdv_metadata.json` - Metadata configuration
- `constraints.json` - Custom constraints (if defined)

## Requirements

See `requirements.txt` for full dependencies. Key packages:
- pandas
- numpy
- scikit-learn
- synthpop (custom-modified version)
- sdv
- plotly (for visualization utilities)

## Notes

- Ensure your input CSV has a proper header row
- Numerical constraints are automatically extracted from the original data
- Generated synthetic data is filtered to stay within original data ranges

## References

- [python-synthpop](https://github.com/NGO-Algorithm-Audit/python-synthpop.git)
- [Synthetic Data Vault (SDV)](https://github.com/sdv-dev/SDV)
- CTAB-GAN+ for tabular data synthesis
