# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 21:56:46 2026

@author: isaac
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV file, skipping the first row (units)
df = pd.read_csv(r"C:\Users\isaac\OneDrive\Documents\McGill\McGill Experiments\waveforms\March 2026\3_23\ionic liquid metal salt dbd\20kHz_35.csv", skiprows=1)

# Rename columns for clarity
df.columns = ['Time (ms)', 'Voltage (V)', 'Current (mA)']

# Convert columns to numeric types
df['Time (ms)'] = pd.to_numeric(df['Time (ms)'], errors='coerce')
df['Voltage (V)'] = pd.to_numeric(df['Voltage (V)'], errors='coerce')
df['Current (mA)'] = pd.to_numeric(df['Current (mA)'], errors='coerce')

# Drop any rows with NaN values
df = df.dropna()

# Assuming time is in seconds (despite label), compute sampling frequency
# Sort by time to ensure order
df = df.sort_values('Time (ms)')
time = (df['Time (ms)'].values)/1000
voltage = (df['Voltage (V)'].values)*1000

# Sampling interval (assume uniform, take mean difference)
dt = np.mean(np.diff(time))
fs = 1 / dt

# Number of samples
N = len(voltage)

# Compute FFT
fft_values = np.fft.fft(voltage)
freqs = np.fft.fftfreq(N, d=dt)

# Only plot positive frequencies
positive_mask = freqs >= 0
freqs = freqs[positive_mask]
magnitude = np.abs(fft_values[positive_mask]) / N  # Normalize

# Plot the spectrum
plt.figure(figsize=(10, 6))
plt.plot(freqs, magnitude, color='blue')  # dB scale
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (V)')
plt.title('FFT of Voltage Spectrum')
plt.grid(True)
plt.xlim(0, fs / 2)  # Up to Nyquist frequency
#plt.xlim(0,2e5)

plt.show()
