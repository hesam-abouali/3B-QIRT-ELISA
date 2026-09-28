"""Faithful re-implementation of 3B-QIRT-ELISA-Peak-Analyzer.py core pipeline."""
import numpy as np, pandas as pd
from scipy.signal import find_peaks as fp, savgol_filter
from numpy.fft import fft, ifft

CH = {'time':'#Digilent WaveForms Oscilloscope Acquisition','Signal655':'Unnamed: 1',
      'Signal605':'Unnamed: 2','SignalAllQDots':'Unnamed: 3','Signal565':'Unnamed: 4'}

def load_scan(path, gains=None):
    df = pd.read_csv(path, encoding='unicode_escape', usecols=list(CH.values())).rename(
        columns={v:k for k,v in CH.items()})
    df = df.drop(df.index[pd.Series(range(0,30))]).astype(float).reset_index(drop=True)
    g = gains or {c:100 for c in ['SignalAllQDots','Signal565','Signal605','Signal655']}
    for c,mult in g.items(): df[c] *= mult
    return df

def baseline_correct(signal, time):
    y_fft = fft(signal); y_fft_filtered = np.copy(y_fft); y_fft_filtered[100:] = 0
    smooth_y = np.real(ifft(y_fft_filtered))
    minima_idx, _ = fp(-smooth_y)
    baseline = (np.interp(time, time[minima_idx], signal[minima_idx]) if len(minima_idx)>0
                else np.median(signal)*np.ones_like(signal))
    return np.maximum(signal - baseline, 0), baseline

def calculate_mad(data):
    m = np.median(data); mad = np.median(np.abs(data-m))
    return mad if mad != 0 else np.std(data)*0.6745

def adaptive_hampel(signal, fold_threshold=20.0, prominence=1.0, min_distance=50):
    signal_std, signal_median = np.std(signal), np.median(signal)
    min_height = max(signal_median + 0.1*signal_std, prominence)
    all_peaks, _ = fp(signal, height=min_height, distance=min_distance, prominence=prominence)
    if len(all_peaks) < 2: return np.array([]), signal, all_peaks, [], {}
    h = signal[all_peaks]; h_med = np.median(h); mad_h = calculate_mad(h)
    if h_med <= 0: h_med = np.mean(h[h>0]) if np.any(h>0) else 1.0
    rsd = mad_h/h_med if h_med>0 else 1.0
    k_eff = (fold_threshold-1)/rsd if rsd>0 else fold_threshold
    thr = h_med*fold_threshold
    mask = h > thr
    outliers, significant = all_peaks[mask], all_peaks[~mask]
    clean = signal.copy(); regions = []
    for idx in outliers:
        w = max(10, int(signal[idx]/h_med*5))
        l, r = max(0, idx-w), min(len(signal)-1, idx+w)
        regions.append((l,r))
        lv, rv = signal[max(0,l-1)], signal[min(len(signal)-1, r+1)]
        clean[l:r+1] = np.linspace(lv, rv, r-l+1)
    info = dict(k_effective=k_eff, rsd=rsd, mad=mad_h, median=h_med,
                fold_threshold=fold_threshold, threshold_ratio=thr,
                threshold_hampel=h_med + k_eff*mad_h)
    return outliers, clean, significant, regions, info

def robust_peak_detection(signal, time, fold=20.0, prominence=None, min_distance=50):
    std, med = np.std(signal), np.median(signal)
    prom = prominence if prominence is not None else max(std, med*0.1)
    outliers, clean, significant, regions, info = adaptive_hampel(signal, fold, prom, min_distance)
    if len(significant) > 0:
        robust_h = max(np.percentile(signal[significant], 50), std)
    else:
        robust_h = med + std
    peaks, props = fp(clean, height=robust_h, prominence=prom, distance=min_distance, wlen=350)
    if len(outliers) > 0 and len(peaks) > 0:
        keep = []
        vp = {'peak_heights':[], 'left_bases':[], 'right_bases':[], 'prominences':[]}
        for i, pk in enumerate(peaks):
            inside = any(l <= pk <= r for l, r in regions)
            # distance to the nearest edge of an excised region, as in the app
            mind = min(min(abs(pk-l), abs(pk-r)) for l, r in regions)
            if not inside and mind > min_distance:
                keep.append(pk)
                for k in vp:
                    if k in props: vp[k].append(props[k][i])
        peaks = np.array(keep)
        props = {k: np.array(v) for k, v in vp.items()}
    return peaks, props, outliers, clean, regions, info

def robust_auc(signal, time, peaks, props):
    if len(peaks) == 0: return np.array([]), 0.0
    aucs = []
    for i in range(len(peaks)):
        l, r = int(props['left_bases'][i]), int(props['right_bases'][i])
        if r - l > 0: aucs.append(np.trapezoid(signal[l:r+1], x=time[l:r+1]))
    a = np.array(aucs)
    if len(a) == 0: return a, 0.0
    t = a[(a >= np.percentile(a,10)) & (a <= np.percentile(a,90))]
    return a, (t.mean() if len(t) else np.median(a))

def analyze(path, channel, fold=20.0, prominence=1.0, min_distance=50, smooth=None, gains=None):
    df = load_scan(path, gains); t = df.time.values
    raw = df[channel].values
    sig = savgol_filter(raw, *smooth) if smooth else raw
    corrected, baseline = baseline_correct(sig, t)
    peaks, props, outliers, clean, regions, info = robust_peak_detection(corrected, t, fold, prominence, min_distance)
    aucs, avg = robust_auc(clean, t, peaks, props)
    return dict(t=t, raw=raw, smoothed=sig, corrected=corrected, baseline=baseline,
                clean=clean, peaks=peaks, props=props, outliers=outliers, regions=regions,
                aucs=aucs, avg_auc=avg, info=info)
