import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

def Q(df, GreenName, RedName, show_plot=False):
    """
    Calculate Q using row-by-row data in `GreenName` vs. `RedName`,
    comparing each row to threshold columns: (GreenName+'Threshold'), (RedName+'Threshold').
    """

    # Compare actual intensities to thresholds
    maskA = df[GreenName] > df[GreenName + 'Threshold']
    maskB = df[RedName]   > df[RedName + 'Threshold']

    # Identify events
    GreenOnlyEvents = df.loc[maskA & (~maskB), GreenName]
    RedOnlyEvents   = df.loc[maskB & (~maskA), RedName]
    GreenEvents     = df.loc[maskA &  maskB,   GreenName]
    RedEvents       = df.loc[maskA &  maskB,   RedName]

    # Shuffle the red mask properly
    # 1) Convert maskB to a NumPy array
    maskB_array = maskB.to_numpy().copy()
    # 2) Shuffle in-place
    np.random.shuffle(maskB_array)
    # 3) Convert back to a Series, reusing the same index
    maskB_shuffled = pd.Series(maskB_array, index=maskB.index)

    # Now define "chance" coincidences as rows where maskA & maskB_shuffled
    GreenChance = df.loc[maskA & maskB_shuffled, GreenName]
    RedChance   = df.loc[maskA & maskB_shuffled, RedName]

    # Compute Q
    var_real_events   = float(len(GreenEvents))
    var_A_events      = float(len(GreenOnlyEvents))
    var_B_events      = float(len(RedOnlyEvents))
    var_chance_events = float(len(GreenChance))

    denom = var_A_events + var_B_events - (var_real_events - var_chance_events)
    if denom == 0:
        print("Warning: denominator for Q calculation is zero. Returning Q=0.")
        Q_value = 0.0
    else:
        Q_value = (var_real_events - var_chance_events) / denom

    print(
        f"[Q] {GreenName}/{RedName}: {var_A_events} A, {var_B_events} B, "
        f"{var_real_events} coincidence, {var_chance_events} chance -> Q = {Q_value:.3f}"
    )

    # Optional: plotting
    if show_plot:
        if (
            len(GreenEvents) > 0 and len(RedEvents) > 0 and
            len(GreenChance) > 0 and len(RedChance) > 0
        ):
            ln_events = np.log(RedEvents / GreenEvents)
            ln_chance = np.log(RedChance / GreenChance)

            textstr = f'Q = {Q_value:.3f}'
            plt.rcParams["font.family"] = "Arial"
            plt.rcParams["font.size"]   = 12
            plt.figure(figsize=(8, 6))

            plt.hist(
                ln_events, bins=60, range=[-3,3], rwidth=0.9, ec='black',
                color='#ff0000', alpha=0.8, label="Real Events"
            )
            plt.hist(
                ln_chance, bins=60, range=[-3,3], rwidth=0.9, ec='black',
                color='#cccccc', alpha=0.5, label="Chance Events"
            )

            plt.text(0.05, 0.90, textstr, transform=plt.gca().transAxes)
            plt.legend(loc='upper right')
            plt.xlabel('Z = ln(I_B / I_A)')
            plt.ylabel('Number of events')
            plt.show()
        else:
            print("Skipping plot due to insufficient data arrays.")

    return Q_value


def process_experiment(experiment, working_path, factor_of_std=3.0, show_plot=False):
    data_file = working_path / experiment
    print(f"[PID {os.getpid()}] Processing: {experiment}")
    data = pd.read_csv(data_file)

    # Group by pinhole location
    grouped_data = data.groupby(['GreenY', 'GreenX', 'RedY', 'RedX'])

    # Compute mean and std for each relevant column
    threshold_df = grouped_data.agg({
        'GreenMax':  ['mean','std'],
        'GreenSum':  ['mean','std'],
        'GreenMean': ['mean','std'],
        'RedMax':    ['mean','std'],
        'RedSum':    ['mean','std'],
        'RedMean':   ['mean','std']
    })

    # Flatten multi-index columns
    threshold_df.columns = [
        'GreenMaxMean','GreenMaxStd',
        'GreenSumMean','GreenSumStd',
        'GreenMeanMean','GreenMeanStd',
        'RedMaxMean','RedMaxStd',
        'RedSumMean','RedSumStd',
        'RedMeanMean','RedMeanStd'
    ]
    threshold_df = threshold_df.reset_index()

    # Create threshold columns: mean + factor_of_std * std
    threshold_df['GreenMaxThreshold']  = threshold_df['GreenMaxMean']  + factor_of_std * threshold_df['GreenMaxStd']
    threshold_df['GreenSumThreshold']  = threshold_df['GreenSumMean']  + factor_of_std * threshold_df['GreenSumStd']
    threshold_df['GreenMeanThreshold'] = threshold_df['GreenMeanMean'] + factor_of_std * threshold_df['GreenMeanStd']

    threshold_df['RedMaxThreshold']    = threshold_df['RedMaxMean']    + factor_of_std * threshold_df['RedMaxStd']
    threshold_df['RedSumThreshold']    = threshold_df['RedSumMean']    + factor_of_std * threshold_df['RedSumStd']
    threshold_df['RedMeanThreshold']   = threshold_df['RedMeanMean']   + factor_of_std * threshold_df['RedMeanStd']

    # Save threshold CSV
    threshold_applied_file = working_path / f'Threshold_used_for_individual_pinhole_{experiment}.csv'
    threshold_df.to_csv(threshold_applied_file, index=False)
    print(f"[PID {os.getpid()}] Thresholds saved to {threshold_applied_file.name}")

    # Merge thresholds back into the main data
    threshold_cols = [
        'GreenY','GreenX','RedY','RedX',
        'GreenMaxThreshold','GreenSumThreshold','GreenMeanThreshold',
        'RedMaxThreshold','RedSumThreshold','RedMeanThreshold'
    ]
    data = pd.merge(
        data,
        threshold_df[threshold_cols],
        on=['GreenY','GreenX','RedY','RedX'],
        how='left'
    )

    # Now call Q for each measurement type
    results = []
    for (g_name, r_name, meas_type) in [
        ('GreenMax',  'RedMax',  'Max'),
        ('GreenSum',  'RedSum',  'Sum'),
        ('GreenMean', 'RedMean', 'Mean')
    ]:
        q_val = Q(data, g_name, r_name, show_plot=show_plot)
        results.append({
            'file': experiment,
            'measurement_type': meas_type,
            'Q_value': q_val
        })

    return results


def main():
    working_path = Path('/Volumes/T7/20250619_results')
    factor_of_std = 2.0  # threshold factor
    show_plot = True    # set to True if you really want many parallel plots

    # Gather all matching CSVs
    csv_files = [
        f for f in os.listdir(working_path)
        if f.endswith('.csv') and f.startswith('timelapse_intensity')
    ]
    print(f"Found {len(csv_files)} CSV files to process in parallel.\n")

    all_q_results = []

    # Use a ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        future_to_file = {
            executor.submit(process_experiment, experiment, working_path, factor_of_std, show_plot): experiment
            for experiment in csv_files
        }

        # As each future completes, retrieve results
        for future in as_completed(future_to_file):
            experiment = future_to_file[future]
            try:
                res = future.result()
                all_q_results.extend(res)
            except Exception as exc:
                print(f"[ERROR] {experiment} generated an exception: {exc}")
            else:
                print(f"[DONE] {experiment} completed.")

    # Convert collected results to DataFrame and save
    results_df = pd.DataFrame(all_q_results)
    output_csv = working_path / 'Q_results.csv'
    results_df.to_csv(output_csv, index=False)
    print(f"\nAll Q values saved to {output_csv.name}")


if __name__ == "__main__":
    main()
