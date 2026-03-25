# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 23:59:28 2026

@author: isaac
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV file, skipping the first row (units)
df = pd.read_csv(r"C:\Users\isaac\OneDrive\Documents\McGill\McGill Experiments\waveforms\March 2026\testarbitrarilychosenfordatamanipulation.csv", skiprows=1)

# Rename columns for clarity
df.columns = ['Time (ms)', 'Voltage (V)', 'Current (mA)']

# Convert columns to numeric types
df['Time (ms)'] = pd.to_numeric(df['Time (ms)'], errors='coerce')
df['Voltage (V)'] = pd.to_numeric(df['Voltage (V)'], errors='coerce')
df['Current (mA)'] = pd.to_numeric(df['Current (mA)'], errors='coerce')

# Drop any rows with NaN values
df = df.dropna()

# Sort by time to ensure order
df = df.sort_values('Time (ms)')
time = df['Time (ms)'].values
voltage = (df['Voltage (V)'].values)*1000

# Compute the first derivative of voltage with respect to time
deriv = np.gradient(voltage, time)

# Plot the derivative
plt.figure(figsize=(10, 6))
plt.plot(time, deriv, color='green')
plt.xlabel('Time (ms)')
plt.ylabel('dV/dt (V/ms)')
plt.title('First Derivative of Voltage Signal')
plt.grid(True)

plt.xlim(0,0.25)

plt.show()
