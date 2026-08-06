"""Dispersion probe: C*(f) of one passive capture at several frequencies.

Usage (from the repo root, same interpreter as the pipeline):
    python dispersion_probe.py "C:/Users/isaac/OneDrive/Documents/McGill/McGill Experiments/waveforms/July 2026/Lissajouswaveformsdifferentmediums.zip"
"""
import sys, io, zipfile
import numpy as np, pandas as pd

zpath = sys.argv[1]
man = pd.read_csv("dbd_surface_charge_report/archive_manifest.csv")
man["level"] = man["level"].astype(str)
rows = man[(man.material == "BMIM_nitrate") & (man.burst_kHz == 20) & (man.level == "75")]
if rows.empty:
    rows = man[(man.material == "BMIM_nitrate") & (man.level == "75")]
member = rows.iloc[0]["member"]
print("capture:", member)

with zipfile.ZipFile(zpath) as z:
    raw = z.read(member)
df = pd.read_csv(io.BytesIO(raw), skiprows=[1]).apply(pd.to_numeric, errors="coerce").dropna()
t = df.iloc[:, 0].values / 1000.0
V = df.iloc[:, 1].values * 1000.0          # 1000x probe -> volts
Q = 100e-9 * df.iloc[:, 3].values           # Cref = 100 nF

# carrier from FFT peak in 100-160 kHz
dt = float(np.median(np.diff(t)))
cen = V - np.mean(V)
sp = np.abs(np.fft.rfft(cen * np.hanning(len(cen))))
fr = np.fft.rfftfreq(len(cen), dt)
band = (fr >= 100e3) & (fr <= 160e3)
fc = float(fr[np.flatnonzero(band)[np.argmax(sp[band])]])
print("carrier: %.1f kHz" % (fc / 1e3))

w = np.hanning(len(t))
def cstar(f):
    X = np.exp(-2j * np.pi * f * t)
    num = np.sum(w * (Q - np.mean(Q)) * X)
    den = np.sum(w * (V - np.mean(V)) * X)
    return num / den if abs(den) > 0 else complex(np.nan, np.nan)

print("%8s %10s %10s %8s" % ("f (kHz)", "C' (pF)", "C'' (pF)", "|V(f)| rel"))
vref = None
for f in [40e3, 60e3, 80e3, 100e3, fc, 140e3, 160e3, 180e3, 2 * fc, 3 * fc]:
    c = cstar(f)
    X = np.exp(-2j * np.pi * f * t)
    vmag = abs(np.sum(w * (V - np.mean(V)) * X))
    vref = vref or abs(np.sum(w * (V - np.mean(V)) * np.exp(-2j * np.pi * fc * t)))
    print("%8.1f %10.2f %10.2f %8.3f" % (f / 1e3, c.real * 1e12, -c.imag * 1e12, vmag / vref))

# endpoint estimate (v1.0-style, full band, time domain)
zc = np.where(np.diff(np.sign(cen)) != 0)[0]
dq = np.abs(np.diff(Q[zc])); dv = np.abs(np.diff(V[zc]))
ok = dv > 0.2 * np.percentile(dv, 90)
print("endpoint |dQ|/|dV| median: %.2f pF" % (np.median(dq[ok] / dv[ok]) * 1e12))
