import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

## TASK 2, CALCULATE, PLOT AND SAVE (e.g. export as .csv) ERRORS from error_angles
df = pd.read_csv('error_angles_random_increased_target_radius.csv')


unperturbed_errors = []

error_angles_masked = np.ma.masked_invalid(df['error_angles'])




unperturbed_errors.extend(error_angles_masked[0:40])
gradual_errors = error_angles_masked[40:60]
sudden_errors = error_angles_masked[120:160]
unperturbed_errors.extend(error_angles_masked[80:120])
unperturbed_errors.extend(error_angles_masked[160:200])


x_axis_data = np.arange(0, len(error_angles_masked), 1)

# Plot error angles with highlighted experimental segments
plt.figure(figsize=(10, 6))
plt.plot(
    error_angles_masked,
    label='Error Angle',
    color='blue',
    linestyle='--',
    marker='o',
    markersize=5
)

# Highlight phases
plt.axvspan(0, 40, color='grey', alpha=0.2, label='No Perturbation')
plt.axvspan(40, 80, color='red', alpha=0.2, label='Gradual Perturbation')
plt.axvspan(80, 120, color='grey', alpha=0.2, label='No Perturbation')
plt.axvspan(120, 160, color='blue', alpha=0.2, label='Sudden Perturbation')
plt.axvspan(160, 200, color='grey', alpha=0.2, label='No Pertubation')

# Add labels, title, and legend
plt.title("Error Angles Over Trials")
plt.xlabel("Trial Number")
plt.ylabel("Error Angle (degrees)")
plt.legend()
plt.show()

# Calculate mean error angle
unperturbed_mean = np.mean(unperturbed_errors)



unperturbed_segment_1 = df["error_angles"][0:40].to_numpy()
unperturbed_segment_2 = df["error_angles"][80:120].to_numpy()
unperturbed_segment_3 = df["error_angles"][160:200].to_numpy()

unperturbed_segments = [unperturbed_segment_1, unperturbed_segment_2, unperturbed_segment_3]

# Calculate motor variability (MV) as Error Variance
for idx, segment  in enumerate(unperturbed_segments):
    unperturbed_mv = np.nanvar(segment)
    print(f"Unperturbed Phase - Motor Variability (MV) for segment {idx}: {unperturbed_mv:.4f}")