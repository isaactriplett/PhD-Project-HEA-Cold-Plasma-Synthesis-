# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 19:04:32 2026

@author: isaac
"""

import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file, skipping the first row (header row)
df = pd.read_csv(r"C:\Users\isaac\OneDrive\Documents\McGill\McGill Experiments\waveforms\March 2026\3_23\ionic liquid metal salt dbd\10kHz_43.csv", skiprows=1)

# The second row becomes the header after skipping, which is units, so rename columns
df.columns = ['Time (ms)', 'Voltage (V)', 'Current (mA)']

# Convert columns to numeric types to ensure they are floats
df['Time (ms)'] = pd.to_numeric(df['Time (ms)'], errors='coerce')
df['Voltage (V)'] = pd.to_numeric(df['Voltage (V)'], errors='coerce')
df['Current (mA)'] = pd.to_numeric(df['Current (mA)'], errors='coerce')

# Optionally, drop any rows with NaN values if conversion failed
df = df.dropna()

# Create the plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot voltage * 1000 on left y-axis
ax1.plot(df['Time (ms)'], df['Voltage (V)'] * 1000, color='blue', label='Voltage')
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Voltage (V)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create twin axis for current / 1000 on right y-axis
ax2 = ax1.twinx()
ax2.plot(df['Time (ms)'], df['Current (mA)'] / 1000, color='red', label='Current')
ax2.set_ylabel('Current (A)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

xlimits=[0.15,0.25]
ax1.set_xlim(xlimits[0],xlimits[1])
ax2.set_xlim(xlimits[0],xlimits[1])

# Add title and show plot
plt.title('Voltage and Current over Time')
fig.tight_layout()
plt.show()