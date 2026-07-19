# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 07:45:51 2026

@author: isaac
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV file, skipping the first row (units)
df = pd.read_csv(r"C:\Users\isaac\OneDrive\Documents\McGill\McGill Experiments\waveforms\June 2026\6_1\40 kHz\Cobalt no dielectric\Cobalt no dielectric_14.csv", skiprows=1)

# Rename columns for clarity
df.columns = ['Time (ms)', 'Voltage (V)', 'Current (A)']

# Convert columns to numeric types
df['Time (ms)'] = pd.to_numeric(df['Time (ms)'], errors='coerce')
df['Voltage (V)'] = pd.to_numeric(df['Voltage (V)'], errors='coerce')
df['Current (A)'] = pd.to_numeric(df['Current (A)'], errors='coerce')

# Drop any rows with NaN values
df = df.dropna()

# Sort by time to ensure order
df = df.sort_values('Time (ms)')
time = (df['Time (ms)'].values)/1000
voltage = (df['Voltage (V)'].values)*1000
current = (df['Current (A)'].values)

# Compute the first derivative of voltage with respect to time
#deriv = np.gradient(voltage, time)

power = current*voltage

"""
# Create the plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot derivative on left y-axis
ax1.plot(time, deriv, color='green', label='dV/dt')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('dV/dt (V/s)', color='green')
ax1.tick_params(axis='y', labelcolor='green')

# Create twin axis for current on right y-axis
ax2 = ax1.twinx()
ax2.plot(time, current, color='red', label='Current')
ax2.set_ylabel('Current (A)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

"""

plt.figure(figsize=(10, 6))
plt.plot(time, power, color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Power (W)')
plt.title('Power Waveform')
plt.grid(True)

plt.xlim(0.000,0.0002)

plt.show()

# Add title and show plot
#plt.title('Product of Current and First Derivative of Voltage')
#fig.tight_layout()

#plt.xlim(0,0.0025)

#plt.show()