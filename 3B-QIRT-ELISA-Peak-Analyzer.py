import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import datetime
from pathlib import Path
from scipy.signal import find_peaks as fp
from scipy.signal import savgol_filter
from numpy.fft import fft, ifft
import warnings
warnings.filterwarnings('ignore')

# Raw scans shipped with the repository (fed-state Rat 1), offered as example data
EXAMPLE_DIR = Path(__file__).parent / "data" / "raw" / "fed_state_Rat1_raw_scans"

st.set_page_config(page_icon="🔬", layout="wide", initial_sidebar_state='auto')
st.title("3B-QIRT-ELISA Peak Analyzer")
st.markdown("### AUC calculation with adaptive Hampel outlier detection")

if 'current_file_index' not in st.session_state:
    st.session_state.current_file_index = 0
if 'all_results' not in st.session_state:
    st.session_state.all_results = []

def calculate_mad(data):
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    if mad == 0:
        mad = np.std(data) * 0.6745
    return mad

def adaptive_hampel_outlier_detection(signal, time, fold_threshold=8.0, 
                                     prominence=0.05, min_distance=10):
    """
    Adaptive Hampel Identifier - Ratio method reframed
    
    Mathematical foundation:
    Standard Hampel: |x - median| > k × MAD
    Adaptive Hampel: x - median > (fold_threshold - 1) × median
                     (x - median) / MAD > k_eff
                     where k_eff = (fold_threshold - 1) / RSD
                     and RSD = MAD / median
    
    References:
    - Hampel, F.R. (1974). J. Am. Stat. Assoc., 69(346), 383-393.
    - Pearson, R.K. (2002). IEEE Trans. Control Syst., 10(1), 55-63.
    """
    signal_std = np.std(signal)
    signal_median = np.median(signal)
    min_height = max(signal_median + 0.1 * signal_std, prominence)
    
    all_peaks, peak_props = fp(signal, height=min_height, 
                               distance=min_distance, prominence=prominence)
    
    if len(all_peaks) < 2:
        return [], signal, all_peaks, [], {}
    
    peak_heights = signal[all_peaks]
    median_height = np.median(peak_heights)
    mad_height = calculate_mad(peak_heights)
    
    if median_height <= 0:
        median_height = np.mean(peak_heights[peak_heights > 0]) if np.any(peak_heights > 0) else 1.0
    
    rsd = mad_height / median_height if median_height > 0 else 1.0
    k_effective = (fold_threshold - 1) / rsd if rsd > 0 else fold_threshold

    threshold_hampel = median_height + k_effective * mad_height
    threshold_ratio = median_height * fold_threshold

    outlier_mask = peak_heights > threshold_ratio
    outlier_indices = all_peaks[outlier_mask]
    significant_peaks = all_peaks[~outlier_mask]
    
    clean_signal = signal.copy()
    outlier_regions = []

    for idx in outlier_indices:
        peak_height = signal[idx]
        width = max(10, int(peak_height / median_height * 5))
        left_boundary = max(0, idx - width)
        right_boundary = min(len(signal) - 1, idx + width)
        outlier_regions.append((left_boundary, right_boundary))

        left_val = signal[max(0, left_boundary - 1)]
        right_val = signal[min(len(signal) - 1, right_boundary + 1)]
        n_points = right_boundary - left_boundary + 1
        clean_signal[left_boundary:right_boundary + 1] = np.linspace(left_val, right_val, n_points)

    hampel_info = {
        'k_effective': k_effective,
        'rsd': rsd,
        'mad': mad_height,
        'median': median_height,
        'fold_threshold': fold_threshold,
        'threshold_hampel': threshold_hampel,
        'threshold_ratio': threshold_ratio,
        'equivalence_check': abs(threshold_hampel - threshold_ratio) / threshold_ratio * 100
    }
    
    return outlier_indices, clean_signal, significant_peaks, outlier_regions, hampel_info

def mad_outlier_detection(signal, time, mad_threshold=3.0, prominence=0.05, min_distance=10):
    signal_std = np.std(signal)
    signal_median = np.median(signal)
    min_height = max(signal_median + 0.1 * signal_std, prominence)
    
    all_peaks, peak_props = fp(signal, height=min_height, distance=min_distance, prominence=prominence)
    
    if len(all_peaks) < 2:
        return [], signal, all_peaks, [], {}
    
    peak_heights = signal[all_peaks]
    median_height = np.median(peak_heights)
    mad = calculate_mad(peak_heights)
    
    if mad == 0:
        mad = np.std(peak_heights) * 0.6745
    
    modified_z_scores = np.abs(peak_heights - median_height) / mad
    outlier_mask = modified_z_scores > mad_threshold
    
    outlier_indices = all_peaks[outlier_mask]
    significant_peaks = all_peaks[~outlier_mask]
    
    clean_signal = signal.copy()
    outlier_regions = []

    for idx in outlier_indices:
        peak_height = signal[idx]
        avg_peak_height = np.median(peak_heights[~outlier_mask]) if np.any(~outlier_mask) else median_height
        width = max(10, int(peak_height / avg_peak_height * 5) if avg_peak_height > 0 else 15)
        left_boundary = max(0, idx - width)
        right_boundary = min(len(signal) - 1, idx + width)
        outlier_regions.append((left_boundary, right_boundary))

        left_val = signal[max(0, left_boundary - 1)]
        right_val = signal[min(len(signal) - 1, right_boundary + 1)]
        n_points = right_boundary - left_boundary + 1
        clean_signal[left_boundary:right_boundary + 1] = np.linspace(left_val, right_val, n_points)
    
    hampel_info = {
        'k_effective': mad_threshold,
        'mad': mad,
        'median': median_height,
        'rsd': mad / median_height if median_height > 0 else 0
    }
    
    return outlier_indices, clean_signal, significant_peaks, outlier_regions, hampel_info

def iqr_outlier_detection(signal, time, iqr_multiplier=1.5, prominence=0.05, min_distance=10):
    signal_std = np.std(signal)
    signal_median = np.median(signal)
    min_height = max(signal_median + 0.1 * signal_std, prominence)
    
    all_peaks, peak_props = fp(signal, height=min_height, distance=min_distance, prominence=prominence)
    
    if len(all_peaks) < 2:
        return [], signal, all_peaks, [], {}
    
    peak_heights = signal[all_peaks]
    q1 = np.percentile(peak_heights, 25)
    q3 = np.percentile(peak_heights, 75)
    iqr = q3 - q1
    
    upper_bound = q3 + iqr_multiplier * iqr
    outlier_mask = peak_heights > upper_bound
    
    outlier_indices = all_peaks[outlier_mask]
    significant_peaks = all_peaks[~outlier_mask]
    
    clean_signal = signal.copy()
    outlier_regions = []

    for idx in outlier_indices:
        peak_height = signal[idx]
        median_height = np.median(peak_heights[~outlier_mask]) if np.any(~outlier_mask) else q3
        width = max(10, int(peak_height / median_height * 5) if median_height > 0 else 15)
        left_boundary = max(0, idx - width)
        right_boundary = min(len(signal) - 1, idx + width)
        outlier_regions.append((left_boundary, right_boundary))

        left_val = signal[max(0, left_boundary - 1)]
        right_val = signal[min(len(signal) - 1, right_boundary + 1)]
        n_points = right_boundary - left_boundary + 1
        clean_signal[left_boundary:right_boundary + 1] = np.linspace(left_val, right_val, n_points)
    
    hampel_info = {
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'median': np.median(peak_heights)
    }
    
    return outlier_indices, clean_signal, significant_peaks, outlier_regions, hampel_info

def robust_peak_detection(signal, time, outlier_method='adaptive_hampel',
                         outlier_threshold=8.0, prominence=None, min_distance=10):
    signal_stats = {
        'mean': np.mean(signal),
        'std': np.std(signal), 
        'median': np.median(signal),
        'p95': np.percentile(signal, 95)
    }
    
    if prominence is None:
        prominence_used = max(signal_stats['std'], signal_stats['median'] * 0.1)
    else:
        prominence_used = prominence
    
    if outlier_method == 'adaptive_hampel':
        outliers, clean_signal, significant_peaks, outlier_regions, hampel_info = \
            adaptive_hampel_outlier_detection(signal, time, outlier_threshold, 
                                             prominence_used, min_distance)
    elif outlier_method == 'mad':
        outliers, clean_signal, significant_peaks, outlier_regions, hampel_info = \
            mad_outlier_detection(signal, time, outlier_threshold, 
                                 prominence_used, min_distance)
    elif outlier_method == 'iqr':
        outliers, clean_signal, significant_peaks, outlier_regions, hampel_info = \
            iqr_outlier_detection(signal, time, outlier_threshold, 
                                 prominence_used, min_distance)
    else:
        outliers, clean_signal, significant_peaks, outlier_regions, hampel_info = \
            adaptive_hampel_outlier_detection(signal, time, outlier_threshold, 
                                             prominence_used, min_distance)
    
    if len(significant_peaks) > 0:
        significant_heights = signal[significant_peaks]
        robust_height = np.percentile(significant_heights, 50)
        robust_height = max(robust_height, signal_stats['std'])
    else:
        robust_height = signal_stats['median'] + signal_stats['std']
    
    peaks, properties = fp(clean_signal, height=robust_height, 
                          prominence=prominence_used, 
                          distance=min_distance, wlen=350)
    
    if len(outliers) > 0 and len(peaks) > 0:
        valid_peaks = []
        valid_peak_props = {'peak_heights': [], 'left_bases': [], 'right_bases': [], 'prominences': []}
        
        for i, peak in enumerate(peaks):
            peak_in_outlier_region = False
            for clean_left, clean_right in outlier_regions:
                if clean_left <= peak <= clean_right:
                    peak_in_outlier_region = True
                    break
            
            min_dist_to_outlier = float('inf')
            for clean_left, clean_right in outlier_regions:
                dist = min(abs(peak - clean_left), abs(peak - clean_right))
                min_dist_to_outlier = min(min_dist_to_outlier, dist)
            
            if not peak_in_outlier_region and min_dist_to_outlier > min_distance:
                valid_peaks.append(peak)
                for key in valid_peak_props.keys():
                    if i < len(properties[key]):
                        valid_peak_props[key].append(properties[key][i])
        
        peaks = np.array(valid_peaks)
        for key in valid_peak_props.keys():
            properties[key] = np.array(valid_peak_props[key])
    
    return peaks, properties, outliers, clean_signal, hampel_info

def calculate_robust_auc(signal, time, peaks, properties, method='trimmed_mean'):
    if len(peaks) == 0:
        return np.array([]), 0
    
    individual_aucs = []
    for i in range(len(peaks)):
        left_base = properties['left_bases'][i]
        right_base = properties['right_bases'][i]
        peak_signal = signal[left_base:right_base+1]
        peak_time = time[left_base:right_base+1]
        
        if len(peak_time) > 1:
            auc = np.trapz(peak_signal, x=peak_time)
            individual_aucs.append(auc)
    
    individual_aucs = np.array(individual_aucs)
    if len(individual_aucs) == 0:
        return individual_aucs, 0
    
    if method == 'trimmed_mean':
        trimmed = individual_aucs[(individual_aucs >= np.percentile(individual_aucs, 10)) & 
                                 (individual_aucs <= np.percentile(individual_aucs, 90))]
        robust_avg = np.mean(trimmed) if len(trimmed) > 0 else np.median(individual_aucs)
    else:
        robust_avg = np.mean(individual_aucs)
    
    return individual_aucs, robust_avg

def calculate_method_quality_metrics(signal, peaks, outliers, individual_aucs):
    metrics = {}
    
    metrics['n_peaks'] = len(peaks)
    metrics['n_outliers'] = len(outliers)
    metrics['outlier_rate'] = len(outliers) / (len(peaks) + len(outliers)) if (len(peaks) + len(outliers)) > 0 else 0
    
    if len(peaks) > 1:
        peak_heights = signal[peaks]
        metrics['peak_height_cv'] = np.std(peak_heights) / np.mean(peak_heights)
        metrics['peak_height_median'] = np.median(peak_heights)
        metrics['peak_height_mad'] = calculate_mad(peak_heights)
    else:
        metrics['peak_height_cv'] = 0
        metrics['peak_height_median'] = 0
        metrics['peak_height_mad'] = 0
    
    if len(individual_aucs) > 0:
        metrics['auc_mean'] = np.mean(individual_aucs)
        metrics['auc_median'] = np.median(individual_aucs)
        metrics['auc_cv'] = np.std(individual_aucs) / np.mean(individual_aucs) if np.mean(individual_aucs) > 0 else 0
    else:
        metrics['auc_mean'] = 0
        metrics['auc_median'] = 0
        metrics['auc_cv'] = 0
    
    if len(outliers) > 0 and len(peaks) > 0:
        outlier_heights = signal[outliers]
        valid_peak_median = np.median(signal[peaks])
        metrics['outlier_size_ratio'] = np.median(outlier_heights) / valid_peak_median if valid_peak_median > 0 else 0
    else:
        metrics['outlier_size_ratio'] = 0
    
    quality_score = 1.0
    if metrics['peak_height_cv'] > 0.5:
        quality_score -= 0.3
    if metrics['n_peaks'] < 3:
        quality_score -= 0.3
    if metrics['outlier_rate'] > 0.2:
        quality_score -= 0.2
    quality_score = max(0, quality_score)
    metrics['quality_score'] = quality_score
    
    return metrics

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📁 Data", 
    "⚙️ Settings", 
    "📊 Processing", 
    "🔍 Analysis",
    "💾 Results"
])

with tab1:
    st.header('Load your data')
    
    def load_data(file):
        data = pd.read_csv(file, encoding='unicode_escape', 
                          usecols=['#Digilent WaveForms Oscilloscope Acquisition','Unnamed: 1','Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']
                         ).rename(columns = {
                             'Unnamed: 3': 'SignalAllQDots',
                             'Unnamed: 4': 'Signal565',
                             'Unnamed: 1': 'Signal655',
                             'Unnamed: 2': 'Signal605',
                             '#Digilent WaveForms Oscilloscope Acquisition': 'time'
                         })
        return data

    def reset_navigation():
        st.session_state.current_file_index = 0
        st.session_state.all_results = []

    use_example = EXAMPLE_DIR.is_dir() and st.checkbox(
        "Use example data: fed-state Rat 1 (24 raw scans from this repository)",
        key="use_example", on_change=reset_navigation)

    if use_example:
        uploaded_files = sorted(EXAMPLE_DIR.glob("scan*.csv"), key=lambda p: int(p.stem[4:]))
        st.info("To reproduce the manuscript's per-scan AUCs, keep the default settings and set "
                "Prominence to 0.05 (QDot 565), 0.5 (QDot 605) and 0.25 (QDot 655) on the Settings tab.")
    else:
        uploaded_files = st.file_uploader("Upload CSV files", type={"csv", "txt"}, accept_multiple_files=True)

    if not uploaded_files:
        st.info("Upload your QIRT-ELISA data files to begin")
        st.stop()

    current_file = uploaded_files[st.session_state.current_file_index]
    st.write(f"**Current:** {st.session_state.current_file_index + 1}/{len(uploaded_files)} - {current_file.name}")

    df = load_data(current_file)
    df.drop(df.index[pd.Series(range(0,30))], inplace=True)
    df = df.astype(float)
    df = df.reset_index(drop=True)
    
    with st.expander("Preview Data"):
        st.dataframe(df.head(20))
    
    st.markdown("### Preamplifier Settings")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        mult1 = st.number_input('All QDots (pA):', min_value=1, value=100)
        df['SignalAllQDots'] *= mult1 
    with col2:
        mult2 = st.number_input('QDot 565 (pA):', min_value=1, value=100)
        df['Signal565'] *= mult2
    with col3:
        mult3 = st.number_input('QDot 605 (pA):', min_value=1, value=100)
        df['Signal605'] *= mult3
    with col4:
        mult4 = st.number_input('QDot 655 (pA):', min_value=1, value=100)
        df['Signal655'] *= mult4

with tab2:
    st.header('Analysis settings')

    outlier_method = st.radio(
        "Select outlier detection method:",
        ['adaptive_hampel', 'iqr'],
        format_func=lambda x: {
            'adaptive_hampel': 'Adaptive Hampel (Fold-change formulation)',
            'iqr': 'IQR (Classic boxplot)'
        }[x]
    )
    
    st.markdown("### Configure thresholds")
    
    if outlier_method == 'adaptive_hampel':
        st.markdown("""
        **Fold-change threshold interpretation:**
        - `8×`: Peaks 8 times larger than median (conservative)
        - `20×`: Peaks 20 times larger (moderate)
        - `30×`: Peaks 30 times larger (aggressive)
        
        """)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            thresh_all = st.slider("All QDots", 5.0, 40.0, 20.0, 0.5)
        with col2:
            thresh_565 = st.slider("QDot 565", 5.0, 40.0, 20.0, 0.5)
        with col3:
            thresh_605 = st.slider("QDot 605", 5.0, 40.0, 20.0, 0.5)
        with col4:
            thresh_655 = st.slider("QDot 655", 5.0, 40.0, 20.0, 0.5)
    
    else:  # iqr
        st.markdown("**IQR Multiplier:** 1.5 = mild outliers, 3.0 = extreme")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            thresh_all = st.slider("All QDots", 1.0, 4.0, 2.0, 0.1)
        with col2:
            thresh_565 = st.slider("QDot 565", 1.0, 4.0, 2.0, 0.1)
        with col3:
            thresh_605 = st.slider("QDot 605", 1.0, 4.0, 2.5, 0.1)
        with col4:
            thresh_655 = st.slider("QDot 655", 1.0, 4.0, 1.5, 0.1)
    
    st.markdown("### Peak detection parameters")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        prom_all = st.number_input("Prominence", 0.01, 5.0, 1.0, 0.01, key="p1")
        dist_all = st.number_input("Min Distance", 5, 100, 50, 5, key="d1")
    with col2:
        prom_565 = st.number_input("Prominence", 0.01, 5.0, 1.0, 0.01, key="p2")
        dist_565 = st.number_input("Min Distance", 5, 100, 50, 5, key="d2")
    with col3:
        prom_605 = st.number_input("Prominence", 0.01, 5.0, 1.0, 0.01, key="p3")
        dist_605 = st.number_input("Min Distance", 5, 100, 50, 5, key="d3")
    with col4:
        prom_655 = st.number_input("Prominence", 0.01, 5.0, 1.0, 0.01, key="p4")
        dist_655 = st.number_input("Min Distance", 5, 100, 50, 5, key="d4")

with tab3:
    st.header('Signal Processing')
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.subheader("All QDots")
        fw1, ov1 = 1, 0
        apply_smooth_all = st.checkbox("Apply smoothing", key="sm_all")
        if apply_smooth_all:
            fw1 = st.number_input("Filter window", 1, 100, 10, key="fw_all")
            ov1 = st.number_input("Poly order", 0, fw1, 1, key="ov_all")
        SignalAllQDots_smoothed = savgol_filter(df.SignalAllQDots, fw1, ov1, mode='interp')
    
    with col2:
        st.subheader("QDot 565")
        fw2, ov2 = 40, 1
        apply_smooth_565 = st.checkbox("Apply smoothing", key="sm_565")
        if apply_smooth_565:
            fw2 = st.number_input("Filter window", 1, 100, 40, key="fw_565")
            ov2 = st.number_input("Poly order", 0, fw2, 1, key="ov_565")
        Signal565_smoothed = savgol_filter(df.Signal565, fw2, ov2, mode='interp')
    
    with col3:
        st.subheader("QDot 605")
        fw3, ov3 = 10, 1
        apply_smooth_605 = st.checkbox("Apply smoothing", key="sm_605")
        if apply_smooth_605:
            fw3 = st.number_input("Filter window", 1, 100, 10, key="fw_605")
            ov3 = st.number_input("Poly order", 0, fw3, 1, key="ov_605")
        Signal605_smoothed = savgol_filter(df.Signal605, fw3, ov3, mode='interp')
    
    with col4:
        st.subheader("QDot 655")
        fw4, ov4 = 20, 1
        apply_smooth_655 = st.checkbox("Apply smoothing", key="sm_655")
        if apply_smooth_655:
            fw4 = st.number_input("Filter window", 1, 100, 20, key="fw_655")
            ov4 = st.number_input("Poly order", 0, fw4, 1, key="ov_655")
        Signal655_smoothed = savgol_filter(df.Signal655, fw4, ov4, mode='interp')
    
    st.markdown("### Baseline correction (FFT-based)")
    
    def baseline_correct(signal, time):
        y_fft = fft(signal)
        y_fft_filtered = np.copy(y_fft)
        y_fft_filtered[100:] = 0
        smooth_y = np.real(ifft(y_fft_filtered))
        minima_idx, _ = fp(-smooth_y)
        if len(minima_idx) > 0:
            baseline = np.interp(time, time[minima_idx], signal[minima_idx])
        else:
            baseline = np.median(signal) * np.ones_like(signal)
        corrected = np.maximum(signal - baseline, 0)
        return corrected, baseline
    
    SignalAllQDots_corrected, _ = baseline_correct(SignalAllQDots_smoothed, df.time.values)
    Signal565_corrected, _ = baseline_correct(Signal565_smoothed, df.time.values)
    Signal605_corrected, _ = baseline_correct(Signal605_smoothed, df.time.values)
    Signal655_corrected, _ = baseline_correct(Signal655_smoothed, df.time.values)
    
    st.success("Processing complete")

with tab4:
    st.header('Peak analysis with adaptive Hampel')
    
    thresholds = {'AllQDots': thresh_all, '565': thresh_565, '605': thresh_605, '655': thresh_655}
    channels_data = {
        'AllQDots': (SignalAllQDots_corrected, prom_all, dist_all),
        '565': (Signal565_corrected, prom_565, dist_565),
        '605': (Signal605_corrected, prom_605, dist_605),
        '655': (Signal655_corrected, prom_655, dist_655)
    }
    
    results = {}
    names = ['All QDots', 'QDot 565 (C-pep)', 'QDot 605 (Glucagon)', 'QDot 655 (Insulin)']
    
    for idx, (ch, (sig, prom, dist)) in enumerate(channels_data.items()):
        st.markdown(f"### {names[idx]}")
        
        peaks, props, outliers, clean_sig, hampel_info = robust_peak_detection(
            sig, df.time.values, outlier_method, thresholds[ch], prom, dist
        )
        
        individual_aucs, avg_auc = calculate_robust_auc(
            clean_sig, df.time.values, peaks, props, 'trimmed_mean'
        )
        
        quality_metrics = calculate_method_quality_metrics(sig, peaks, outliers, individual_aucs)
        
        results[ch] = {
            'peaks': peaks, 'properties': props, 'outliers': outliers,
            'all_aucs': individual_aucs, 'avg_auc': avg_auc,
            'quality_metrics': quality_metrics, 'hampel_info': hampel_info
        }
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.time, y=sig, name='Signal', line=dict(color='blue', width=1.5)))
        
        if len(peaks) > 0:
            fig.add_trace(go.Scatter(x=df.time.iloc[peaks], y=sig[peaks], 
                                    mode='markers', name='Valid', marker=dict(color='green', size=10)))
        if len(outliers) > 0:
            fig.add_trace(go.Scatter(x=df.time.iloc[outliers], y=sig[outliers],
                                    mode='markers', name='Outliers', marker=dict(color='red', size=12, symbol='x')))
        
        fig.update_layout(height=350, showlegend=True, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Peaks", len(peaks))
        with col2:
            st.metric("Outliers", len(outliers))
        with col3:
            st.metric("Avg AUC", f"{avg_auc:.2f}")
        with col4:
            auc_cv = quality_metrics['auc_cv']
            if auc_cv < 0.2:
                cv_indicator = "Excellent"
            elif auc_cv < 0.4:
                cv_indicator = "Good"
            else:
                cv_indicator = "Variable"
            st.metric("AUC CV", f"{auc_cv:.3f}", cv_indicator)
        with col5:
            if outlier_method == 'adaptive_hampel':
                st.metric("k_eff", f"{hampel_info.get('k_effective', 0):.2f}")
            else:
                st.metric("Quality", f"{quality_metrics['quality_score']:.2f}")
        
        st.markdown("---")
    
    st.session_state.current_results = results

with tab5:
    st.header('Results & export')
    
    if 'current_results' not in st.session_state:
        st.warning("Complete analysis first")
        st.stop()
    
    results = st.session_state.current_results
    
    st.write(f"**File:** {current_file.name}")
    st.write(f"**Method:** {outlier_method}")
    
    summary = []
    for ch, name in zip(['AllQDots', '565', '605', '655'], 
                       ['All QDots', 'QDot 565', 'QDot 605', 'QDot 655']):
        hampel_info = results[ch].get('hampel_info', {})
        quality = results[ch].get('quality_metrics', {})
        
        all_aucs = results[ch]['all_aucs']
        auc_std = np.std(all_aucs) if len(all_aucs) > 0 else 0
        avg_auc = results[ch]['avg_auc']
        auc_cv = (auc_std / avg_auc) if avg_auc > 0 else 0
        
        row = {
            'Channel': name,
            'Peaks': len(results[ch]['peaks']),
            'Outliers': len(results[ch]['outliers']),
            'Avg AUC': f"{avg_auc:.2f}",
            'AUC StDev': f"{auc_std:.2f}",
            'AUC CV': f"{auc_cv:.3f}",
            'Quality': f"{quality.get('quality_score', 0):.2f}"
        }
        
        if outlier_method == 'adaptive_hampel':
            row['k_eff'] = f"{hampel_info.get('k_effective', 0):.2f}"
            row['RSD'] = f"{hampel_info.get('rsd', 0):.3f}"
        
        summary.append(row)
    
    st.dataframe(pd.DataFrame(summary), use_container_width=True)
    
    st.info("**AUC CV (Coefficient of Variation):** Measures the variability of individual peak AUCs. Lower values indicate more consistent peaks. Typical values: <0.2 = excellent, 0.2-0.4 = good, >0.4 = high variability.")
    
    st.markdown("### File navigation")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.current_file_index > 0:
            if st.button("⬅️ Previous"):
                if current_file.name not in [r['file'] for r in st.session_state.all_results]:
                    st.session_state.all_results.append({
                        'file': current_file.name,
                        'method': outlier_method,
                        'results': results
                    })
                st.session_state.current_file_index -= 1
                st.rerun()
    
    with col2:
        st.write(f"**{st.session_state.current_file_index + 1}/{len(uploaded_files)}**")
    
    with col3:
        if st.session_state.current_file_index < len(uploaded_files) - 1:
            if st.button("Next ➡️"):
                if current_file.name not in [r['file'] for r in st.session_state.all_results]:
                    st.session_state.all_results.append({
                        'file': current_file.name,
                        'method': outlier_method,
                        'results': results
                    })
                st.session_state.current_file_index += 1
                st.rerun()
        else:
            if st.button("Finish"):
                if current_file.name not in [r['file'] for r in st.session_state.all_results]:
                    st.session_state.all_results.append({
                        'file': current_file.name,
                        'method': outlier_method,
                        'results': results
                    })
                st.success("All files processed!")
    
    if len(st.session_state.all_results) > 0:
        st.markdown("---")
        st.markdown("### Export results")
        
        st.markdown("#### Summary by channel (long format)")
        export_data = []
        for item in st.session_state.all_results:
            for ch, ch_name in zip(['AllQDots', '565', '605', '655'],
                                  ['All QDots', 'QDot 565', 'QDot 605', 'QDot 655']):
                result = item['results'][ch]
                hampel_info = result.get('hampel_info', {})
                quality = result.get('quality_metrics', {})
                auc_std = np.std(result['all_aucs']) if len(result['all_aucs']) > 0 else 0
                avg_auc = result['avg_auc']
                
                row = {
                    'File': item['file'],
                    'Method': item['method'],
                    'Channel': ch_name,
                    'Peaks': len(result['peaks']),
                    'Outliers': len(result['outliers']),
                    'Avg_AUC': avg_auc,
                    'AUC_StDev': auc_std,
                    'AUC_CV': (auc_std / avg_auc) if avg_auc > 0 else 0,
                    'Quality_Score': quality.get('quality_score', 0),
                    'Peak_CV': quality.get('peak_height_cv', 0)
                }
                
                if item['method'] == 'adaptive_hampel':
                    row['k_effective'] = hampel_info.get('k_effective', 0)
                    row['RSD'] = hampel_info.get('rsd', 0)
                    row['Fold_Threshold'] = hampel_info.get('fold_threshold', 0)
                
                export_data.append(row)
        
        export_df = pd.DataFrame(export_data)
        csv1 = export_df.to_csv(index=False)
        
        st.download_button(
            "Download summary CSV",
            csv1,
            f"qirt_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            key="export1"
        )
        
        with st.expander("Preview Summary CSV"):
            st.dataframe(export_df, use_container_width=True)
        
        st.markdown("#### All metrics by file (wide format)")
        wide_data = []
        for item in st.session_state.all_results:
            row = {'File': item['file'], 'Method': item['method']}
            
            for ch, ch_name in zip(['AllQDots', '565', '605', '655'],
                                  ['AllQDots', 'QD565', 'QD605', 'QD655']):
                result = item['results'][ch]
                auc_std = np.std(result['all_aucs']) if len(result['all_aucs']) > 0 else 0
                avg_auc = result['avg_auc']
                quality = result.get('quality_metrics', {})
                
                row[f'{ch_name}_Peaks'] = len(result['peaks'])
                row[f'{ch_name}_Outliers'] = len(result['outliers'])
                row[f'{ch_name}_Avg_AUC'] = avg_auc
                row[f'{ch_name}_AUC_StDev'] = auc_std
                row[f'{ch_name}_AUC_CV'] = (auc_std / avg_auc) if avg_auc > 0 else 0
                row[f'{ch_name}_Quality'] = quality.get('quality_score', 0)
                
                if item['method'] == 'adaptive_hampel':
                    hampel_info = result.get('hampel_info', {})
                    row[f'{ch_name}_k_eff'] = hampel_info.get('k_effective', 0)
                    row[f'{ch_name}_RSD'] = hampel_info.get('rsd', 0)
            
            wide_data.append(row)
        
        wide_df = pd.DataFrame(wide_data)
        csv2 = wide_df.to_csv(index=False)
        
        st.download_button(
            "Download wide format CSV",
            csv2,
            f"qirt_wide_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            key="export2"
        )
        
        with st.expander("Preview Wide Format CSV"):
            st.dataframe(wide_df, use_container_width=True)
        
        st.markdown("#### All individual peak AUCs (detailed)")
        detailed_data = []
        for item in st.session_state.all_results:
            for ch, ch_name in zip(['AllQDots', '565', '605', '655'],
                                  ['All QDots', 'QDot 565', 'QDot 605', 'QDot 655']):
                result = item['results'][ch]
                for peak_idx, auc_val in enumerate(result['all_aucs'], 1):
                    detailed_data.append({
                        'File': item['file'],
                        'Method': item['method'],
                        'Channel': ch_name,
                        'Peak_Number': peak_idx,
                        'AUC': auc_val,
                        'Peak_Index': result['peaks'][peak_idx-1] if peak_idx-1 < len(result['peaks']) else None
                    })
        
        detailed_df = pd.DataFrame(detailed_data)
        csv3 = detailed_df.to_csv(index=False)
        
        st.download_button(
            "📥 Download detailed Ppeak AUCs CSV",
            csv3,
            f"qirt_detailed_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            key="export3"
        )
        
        with st.expander("Preview Detailed CSV"):
            st.dataframe(detailed_df, use_container_width=True)
        
        st.success("All export options ready! Download the formats you need.")