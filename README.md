# 3B-QIRT-ELISA Peak Analyzer

[![Paper](https://img.shields.io/badge/JOURNAL-2026-blue)](https://doi.org/...)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
![Python 3.10–3.13](https://img.shields.io/badge/python-3.10--3.13-blue)

A Streamlit app that detects peaks and calculates their areas (AUC) in three-biomarker measurements of the QIRT-ELISA platform: 
QDot 565 (C-peptide), QDot 605 (glucagon), and QDot 655 (insulin). 
Spurious outliers are identified and screened out with an adaptive Hampel (fold-change) rule before integration.

This repository also contains the raw recordings of one experiment (fed-state Rat 1, 24 scans)
and a notebook that recomputes the manuscript's per-scan peak areas from them (data that has been used for the two-point calibration of the concentration profile of Rat 1 in the fed-state experiment, and subsequent feature extraction from those concentration profiles).

<!-- Live app: the Streamlit Community Cloud link here after deploying -->

## Run the app

**Online:** open the live app and tick **Use example data** on the Data tab to load the 24 raw
scans in `data/raw/fed_state_Rat1_raw_scans/`.

**Locally:**

```bash
pip install -r requirements.txt
streamlit run 3B-QIRT-ELISA-Peak-Analyzer.py
```

Recordings of Digilent WaveForms software (CSV exports) can be uploaded on the Data tab.

## Contents

```
3B-QIRT-ELISA-Peak-Analyzer.py            Streamlit app, main app of the package
requirements.txt                          package versions used (app and notebook)
data/raw/fed_state_Rat1_raw_scans/        24 raw scans as exported by Digilent WaveForms (unmodified)
data/processed/                           the app's exported results for the 24 scans (Download wide format CSV)
analysis/pipeline.py                      the app's processing chain as plain functions (no interface)
analysis/peak_analysis_walkthrough.ipynb  step-by-step walkthrough with plots; reproduces the app's export
```

File formats and column definitions are in [data/README.md](data/README.md).

## How the peak areas are calculated

A step-by-step version, with a plot at each step, is in [`analysis/peak_analysis_walkthrough.ipynb`](analysis/peak_analysis_walkthrough.ipynb).

Each scan is processed one channel at a time:

1. **Load and scale** The WaveForms preamble is discarded, and each
   channel is multiplied by its preamplifier gain setting (100) to give the photocurrent
   $`u(t)`$ in pA.
2. **Smooth (applied by default)** A Savitzky–Golay filter gives $`\hat u`$, with (window, order) = (40, 1)
   for QDot 565, (10, 1) for QDot 605, and (20, 1) for QDot 655; the window and order can be changed
   on the app's Processing tab.
3. **Remove the baseline** FFT coefficients of $`\hat u`$ from index 100 upward are set to zero.
   The local minima of the resulting low-pass curve are the anchor points, and the baseline
   $`\beta(t)`$ interpolates $`\hat u`$ linearly through them. The corrected signal is
   $`u^\star(t) = \max(\hat u(t) - \beta(t),\ 0)`$.
4. **Screen out spurious outlier peaks (adaptive Hampel)** Candidate peaks of height $`h_i`$ are found
   on $`u^\star`$ with `scipy.signal.find_peaks`, using minimum height
   $`\max(\mathrm{median}(u^\star) + 0.1\,\mathrm{sd}(u^\star),\ p_\mathrm{prom})`$, prominence
   $`p_\mathrm{prom}`$ and minimum separation $`n_\mathrm{min} = 50`$ samples. A candidate is an
   outlier when $`h_i > \Lambda\, h_\mathrm{med}`$, where $`h_\mathrm{med}`$ is the median candidate
   height and $`\Lambda = 20`$. This is the one-sided Hampel test
   $`h_i - h_\mathrm{med} > k_\mathrm{eff}\,\mathrm{MAD}_h`$ with
   $`k_\mathrm{eff} = (\Lambda - 1)/\mathrm{RSD}`$ and $`\mathrm{RSD} = \mathrm{MAD}_h / h_\mathrm{med}`$.
5. **Excise outliers.** Around each outlier apex, the samples within
   $`w_\mathrm{rep} = \max(10,\ \lfloor 5\,h_i / h_\mathrm{med} \rfloor)`$ are replaced by a
   straight line joining the samples just outside that window.
6. **Find the peaks to integrate** `find_peaks` runs again on the repaired signal with the
   same $`p_\mathrm{prom}`$ and $`n_\mathrm{min}`$, a 350-sample prominence window (`wlen`), and a
   minimum height equal to the larger of $`\mathrm{sd}(u^\star)`$ and the median height of the
   non-outlier candidates. Peaks inside an excised region, or within $`n_\mathrm{min}`$ samples of
   one, are dropped.
7. **Integrate** Each peak's area $`A_i`$ (pA·s) is the trapezoidal integral of the repaired
   signal between the left and right bases that `find_peaks` reports for that peak.
8. **Summarize the scan** The scan value $`\bar{A}`$ is the mean of the $`A_i`$ that lie between
   their 10th and 90th percentiles.

The same settings were used for every scan in the manuscript: $`p_\mathrm{prom}`$ = 0.05
(QDot 565), 0.5 (QDot 605) and 0.25 (QDot 655), with everything else at the app defaults
($`\Lambda`$ = 20, $`n_\mathrm{min}`$ = 50, gain 100, smoothing as in step 2). The Settings tab
also offers a classical IQR outlier rule, which the manuscript does not use.

**Time course** Each channel's scan values are divided by their mean over all scans, $`F_0`$,
and consecutive pairs of scans (1, 2), (3, 4), … are averaged into 12 points at 2.5-min
spacing. Insulin scan 14 has no detectable peaks, so its 2.5-min point uses scan 13 alone. The
last two minutes of this recording were not saved, so the 30-min point repeats the 27.5-min values.

## Reproducing the manuscript values

```bash
pip install -r requirements.txt notebook
jupyter notebook analysis/peak_analysis_walkthrough.ipynb   # opens it in the browser
jupyter execute analysis/peak_analysis_walkthrough.ipynb    # or runs it without opening it
```

The notebook recomputes the peak counts, outlier counts and peak areas of all 24 scans for the
three hormones from the raw files, checks them against the app's export
`data/processed/Fed_state_Rat1_app_output.csv`, and computes the normalized 2.5-min time course.
GitHub runs it after every push.

`analysis/pipeline.py` is the app's processing chain as plain functions, without the interface.
The app itself gives the same values: tick **Use example data** on the Data tab and set the
prominences listed above on the Settings tab.




---

## Contact

**Hesam Abouali**\
[[hesam.abouali@uwaterloo.ca](mailto:hesam.abouali@uwaterloo.ca)] Department of Electrical & Computer Engineering\
University of Waterloo, Waterloo, ON, Canada

**Mahla Poudineh**\
[[mahla.poudineh@uwaterloo.ca](mailto:mahla.poudineh@uwaterloo.ca)] Department of Electrical & Computer Engineering\
University of Waterloo, Waterloo, ON, Canada

---

## Acknowledgments

This work was supported by the Canadian Institutes of Health Research (CIHR) and the Natural Sciences and Engineering Research Council of Canada (NSERC).

---
