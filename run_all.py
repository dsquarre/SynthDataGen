from metadata_manager import generate_metadata, interactive_edit_metadata, save_as_sdv_json
import pandas as pd
from synth import run_synthpop
from sdv_all import run_sdv_models
from ctab_gan_plus import run_ctab_gan_plus
from metrics import evaluate_all
from ros_augmentation import apply_ros_with_all_constraints, regenerate_until_target
from custom_constraints import interactive_add_constraints, load_constraints, save_constraints

def main():
    print("Loading data.csv")
    df = pd.read_csv('data.csv')
    try:
        df.drop(columns=df.columns[0], inplace=True)
    except Exception:
        pass
    df.columns = df.columns.str.strip()

    print("Generating metadata...")
    metadata = generate_metadata(df)

    print("You can edit column types now. Use index and 'n' or 'c' (e.g. '3 n'). Type 'done' when finished.")
    metadata = interactive_edit_metadata(metadata)

    print("Saving SDV-compatible metadata to sdv_metadata.json")
    save_as_sdv_json(metadata, path='sdv_metadata.json')

    print("\nWould you like to add custom constraints? (e.g., w/c = Water/Cement, Volume = L*B*H)")
    add_constraints = input("Add custom constraints? (y/n): ").strip().lower()
    custom_constraints = []
    if add_constraints == 'y':
        custom_constraints = interactive_add_constraints()
        if custom_constraints:
            save_constraints(custom_constraints)

    print("\nRunning synthpop models (CART, GaussianCopula)")
    synth_outputs = run_synthpop(df=df, metadata=metadata, cart_samples=2000, gc_samples=2000)

    print("Running SDV single-table models")
    sdv_outputs = run_sdv_models(df=df, sdv_metadata_path='sdv_metadata.json', num_rows=2000)

    print("Running CTAB-GAN+ model")
    try:
        ctab_csv = run_ctab_gan_plus(df=df, num_samples=500)
        print(f"CTAB-GAN+ output: {ctab_csv}")
    except Exception as e:
        print(f"CTAB-GAN+ failed: {e}")
        ctab_csv = None

    # Collect all outputs for metrics
    outputs = {}
    for k, v in synth_outputs.items():
        outputs[f'synthpop_{k}'] = v
    for k, v in sdv_outputs.items():
        outputs[f'sdv_{k}'] = v
    if ctab_csv:
        outputs['ctab_gan_plus'] = ctab_csv

    print("Generating comparative metrics report for all models...")
    results = evaluate_all(df, outputs, metadata)
    print("All done. Outputs and reports written to current directory.")

    # Apply Random Oversampling with constraint validation to synthetic data
    print("\nApplying Random Oversampling (ROS) with failsafe to synthetic datasets...")
    print(f"Target size per model: {len(df) * 2} rows (2× real data)\n")
    
    target_col = df.columns[-1]
    target_size = len(df) * 2
    
    for name, synth_data in outputs.items():
        try:
            print(f"Processing {name}...")
            
            # First attempt: standard ROS + constraints
            ros_df, metadata = apply_ros_with_all_constraints(synth_data, df, custom_constraints=custom_constraints, target_column=target_col)
            
            # Check if failsafe triggered (>50% data rejected)
            if not metadata['passed_failsafe']:
                print(f"WARNING: {metadata['rejection_rate']*100:.1f}% of data was rejected!")
                print(f"  Triggering failsafe: regenerating to reach target size...")
                
                # Regenerate until target is reached
                ros_df, failsafe_meta = regenerate_until_target(
                    synth_data, df, 
                    custom_constraints=custom_constraints,
                    target_column=target_col,
                    target_size=target_size,
                    max_regenerations=5,
                    verbose=True
                )
                
                # Save failsafe metadata
                with open(f"{name}_failsafe_report.txt", 'w') as f:
                    f.write(f"Failsafe Report for {name}\n")
                    f.write(f"{'='*50}\n\n")
                    f.write(f"Regeneration attempts: {failsafe_meta['regeneration_attempts']}\n")
                    f.write(f"Target size: {failsafe_meta['target_size']} rows\n")
                    f.write(f"Final rows: {failsafe_meta['final_rows']} rows\n")
                    f.write(f"Target met: {'YES ✓' if failsafe_meta['target_met'] else 'NO ✗'}\n\n")
                    f.write("Regeneration History:\n")
                    for hist in failsafe_meta['regeneration_history']:
                        f.write(f"  Attempt {hist['attempt']}: {hist['input_rows']} → {hist['output_rows']} rows ({hist['rejection_rate']*100:.1f}% rejected)\n")
                print(f"  Failsafe report saved to: {name}_failsafe_report.txt")
            else:
                print(f"  ✓ Passed: {len(synth_data)} → {metadata['initial_rows']} (ROS) → {ros_df.shape[0]} (after validation)")
                print(f"  Rejection rate: {metadata['rejection_rate']*100:.1f}%")
            
            ros_file = f"{name}_ros_data.xlsx"
            ros_df.to_excel(ros_file, index=False)
            print(f"  Saved to: {ros_file}\n")
            
        except Exception as e:
            print(f"  ✗ ROS failed for {name}: {e}\n")

if __name__ == '__main__':
    main()
