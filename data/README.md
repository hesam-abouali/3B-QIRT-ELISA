# Data

## Raw scans: `raw/fed_state_Rat1_raw_scans/scan1.csv` … `scan24.csv`

The 24 consecutive scans of the fed-state Rat 1 experiment, exactly as exported by Digilent
WaveForms (oscilloscope acquisition, device "Discovery3"). Lines 1–16 of each file are a
preamble with the device, the acquisition date and time, the sample rate (215.579 Hz). 
The data columns follow:

| Column | Signal | Name in the app |
|---|---|---|
| Time (s) | acquisition time | `time` |
| Channel 1 (V) | QDot 655: insulin | `Signal655` |
| Channel 2 (V) | QDot 605: glucagon | `Signal605` |
| Channel 3 (V) | all QDots | `SignalAllQDots-High-pass filter` |
| Channel 4 (V) | QDot 565: C-peptide | `Signal565` |

The app discards the preamble and the first 15 samples, then multiplies each channel by its
preamplifier gain setting (100 for every channel in this dataset) to give photocurrent in pA.

## Processed results: `processed/Fed_state_Rat1_app_output.csv`

The app's **Download wide format CSV** export for the 24 scans, one row per scan, produced with the
settings given in the main README. `File` names the raw scan and `Method` the outlier rule
(`adaptive_hampel`). Each channel (`AllQDots`, `QD565` C-peptide, `QD605` glucagon, `QD655`
insulin) has eight columns:

| Column | Content |
|---|---|
| `<channel>_Peaks` | number of peaks integrated |
| `<channel>_Outliers` | number of candidate peaks flagged as outliers and excised |
| `<channel>_Avg_AUC` | trimmed-mean peak area of the scan, Ā (pA·s); 0 when no peaks were detected |
| `<channel>_AUC_StDev` | standard deviation of the individual peak areas (pA·s) |
| `<channel>_AUC_CV` | `AUC_StDev` / `Avg_AUC` |
| `<channel>_Quality` | the app's quality score, from 0 to 1 |
| `<channel>_k_eff` | effective Hampel multiplier, (Λ − 1) / RSD |
| `<channel>_RSD` | MAD / median of the candidate peak heights |

The `AllQDots` channel is processed with the app defaults and is not used in the manuscript.
Insulin scan 14 has no detectable peaks, so its `QD655_Avg_AUC` is 0.

F<sub>0</sub> (the mean Ā of a hormone over all scans), the normalized values Ā/F<sub>0</sub> and the
2.5-min points (means of scans 1–2, 3–4, …) are computed from `Avg_AUC` by the notebook
`analysis/peak_analysis_walkthrough.ipynb`.
The last two minutes of this recording were not saved, so the 30-min point repeats the 27.5-min values.
