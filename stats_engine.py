"""
MaxQuant LC-MS/MS Proteomics - Statistical Analysis Module v2
==============================================================
Vectorized differential abundance, time-course analysis, modeling.
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


# ── Vectorized Differential Abundance (v2: ~50-100x faster) ──

def differential_abundance(df, cols_a, cols_b, label_a='A', label_b='B'):
    """Vectorized log2FC and Welch's t-test between two groups."""
    A = df[cols_a].replace(0, np.nan).values.astype(float)
    B = df[cols_b].replace(0, np.nan).values.astype(float)

    log2A = np.log2(np.where(A > 0, A, np.nan))
    log2B = np.log2(np.where(B > 0, B, np.nan))

    mean_a = np.nanmean(log2A, axis=1)
    mean_b = np.nanmean(log2B, axis=1)
    log2fc = mean_a - mean_b

    # Vectorized Welch's t-test
    n_a = np.sum(~np.isnan(log2A), axis=1)
    n_b = np.sum(~np.isnan(log2B), axis=1)
    var_a = np.nanvar(log2A, axis=1, ddof=1)
    var_b = np.nanvar(log2B, axis=1, ddof=1)

    # Safe division
    se = np.sqrt(np.where(n_a > 0, var_a / np.maximum(n_a, 1), 0) +
                 np.where(n_b > 0, var_b / np.maximum(n_b, 1), 0))
    t_stat = np.where(se > 0, (mean_a - mean_b) / se, 0)

    # Welch-Satterthwaite df
    num = (var_a / np.maximum(n_a, 1) + var_b / np.maximum(n_b, 1)) ** 2
    denom = np.where(n_a > 1, (var_a / np.maximum(n_a, 1))**2 / (n_a - 1), 0) + \
            np.where(n_b > 1, (var_b / np.maximum(n_b, 1))**2 / (n_b - 1), 0)
    welch_df = np.where(denom > 0, num / denom, 1)
    welch_df = np.maximum(welch_df, 1)

    pvals = 2 * stats.t.sf(np.abs(t_stat), welch_df)
    # Set p=1 where we can't compute (insufficient data)
    insufficient = (n_a < 2) | (n_b < 2)
    pvals = np.where(insufficient, 1.0, pvals)
    pvals = np.where(np.isnan(pvals), 1.0, pvals)

    result = pd.DataFrame({
        'Protein IDs': df.get('Protein IDs', pd.Series(range(len(df)))),
        'Majority protein IDs': df.get('Majority protein IDs', ''),
        'Protein names': df.get('Protein names', ''),
        'description': df.get('description', ''),
        'Gene names': df.get('Gene names', ''),
        'log2FC': log2fc,
        'pvalue': pvals,
        '-log10p': -np.log10(np.maximum(pvals, 1e-300)),
    }, index=df.index)
    return result


# ── Time-Course Analysis (v2: NEW) ───────────────────────────

def timecourse_analysis(df, group_cols, group_names, baseline_group=None):
    """Compute fold-changes and trends across ordered time points.
    
    Args:
        df: DataFrame with quantification columns
        group_cols: dict {group_name: [col1, col2, ...]}
        group_names: ordered list of group names (e.g. ['Day0','Day3','Day7'])
        baseline_group: name of baseline group (default: first)
    
    Returns:
        DataFrame with mean, pct_of_baseline, log2FC, pvalue, trend per protein
    """
    if baseline_group is None:
        baseline_group = group_names[0]

    # Compute group means
    means = {}
    for g in group_names:
        vals = df[group_cols[g]].replace(0, np.nan).values.astype(float)
        means[g] = np.nanmean(vals, axis=1)

    baseline = means[baseline_group]
    results = pd.DataFrame(index=df.index)

    for g in group_names:
        results[f'mean_{g}'] = means[g]
        results[f'pct_{g}'] = np.where(baseline > 0, means[g] / baseline * 100, np.nan)

    # Compute FC and p-value vs baseline for each non-baseline group
    for g in group_names:
        if g == baseline_group:
            results[f'log2FC_{g}'] = 0.0
            results[f'pval_{g}'] = 1.0
            continue
        fc = np.where((baseline > 0) & (means[g] > 0),
                      np.log2(means[g] / baseline), np.nan)
        results[f'log2FC_{g}'] = fc

        # Vectorized t-test vs baseline
        A = np.log2(np.where(df[group_cols[baseline_group]].replace(0, np.nan).values > 0,
                             df[group_cols[baseline_group]].replace(0, np.nan).values, np.nan))
        B = np.log2(np.where(df[group_cols[g]].replace(0, np.nan).values > 0,
                             df[group_cols[g]].replace(0, np.nan).values, np.nan))
        n_a = np.sum(~np.isnan(A), axis=1)
        n_b = np.sum(~np.isnan(B), axis=1)
        pvals = np.ones(len(df))
        for i in range(len(df)):
            a_vals = A[i][~np.isnan(A[i])]
            b_vals = B[i][~np.isnan(B[i])]
            if len(a_vals) >= 2 and len(b_vals) >= 2:
                _, pvals[i] = stats.ttest_ind(a_vals, b_vals, equal_var=False)
        results[f'pval_{g}'] = pvals

    # Trend classification: requires BOTH fold-change AND statistical significance (p < 0.05)
    last_g = group_names[-1]
    fc_last = results[f'log2FC_{last_g}']
    pv_last = results[f'pval_{last_g}']
    sig = pv_last < 0.05  # statistical significance gate
    results['trend'] = np.where((fc_last < -0.5) & sig, 'Degrading',
                       np.where((fc_last > 0.5) & sig, 'Increasing', 'Stable'))
    results.loc[fc_last.isna(), 'trend'] = 'Stable'

    # Copy metadata
    for col in ['Protein IDs', 'Majority protein IDs', 'description',
                'Gene names', 'Fasta headers', 'allergen_code', 'label']:
        if col in df.columns:
            results[col] = df[col].values

    return results


# ── Multiple Testing Correction ──────────────────────────────

def benjamini_hochberg(pvalues, fdr=0.05):
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    pvals = np.array(pvalues)
    n = len(pvals)
    if n == 0:
        return np.array([])
    sorted_idx = np.argsort(pvals)
    sorted_pvals = pvals[sorted_idx]
    adjusted = np.zeros(n)
    adjusted[sorted_idx[-1]] = sorted_pvals[-1]
    for i in range(n - 2, -1, -1):
        adjusted[sorted_idx[i]] = min(
            adjusted[sorted_idx[i + 1]],
            sorted_pvals[i] * n / (i + 1)
        )
    return np.minimum(adjusted, 1.0)


def s0_significance(log2fc, pvalue, s0=0.1, fdr=0.05):
    """s0-based significance thresholding (Giai Gianetto et al. 2016)."""
    t_stat = stats.t.ppf(1 - fdr / 2, df=4)
    fc_abs = np.abs(log2fc)
    neg_log10p = -np.log10(np.maximum(pvalue, 1e-300))
    threshold = t_stat * np.sqrt(1 + s0**2 / (fc_abs**2 + 1e-10))
    return neg_log10p > threshold


def classify_significance(diff_df, fc_thresh=1.0, pval_thresh=0.05):
    """Classify proteins as up/down/non-significant."""
    diff_df = diff_df.copy()
    diff_df['significant'] = (
        (abs(diff_df['log2FC']) > fc_thresh) &
        (diff_df['pvalue'] < pval_thresh)
    )
    diff_df['direction'] = 'NS'
    diff_df.loc[diff_df['significant'] & (diff_df['log2FC'] > 0), 'direction'] = 'Up'
    diff_df.loc[diff_df['significant'] & (diff_df['log2FC'] < 0), 'direction'] = 'Down'
    return diff_df


# ── Dimensionality Reduction ─────────────────────────────────

def run_pca(data_matrix, n_components=2):
    """Run PCA on protein quantification matrix."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data_matrix.fillna(0).T)
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(scaled)
    return coords, pca.explained_variance_ratio_


# ── Predictive Modeling ──────────────────────────────────────

def train_classifier(X, y, model_type='svm', cv='loo'):
    """Train a classifier for sample group prediction."""
    if model_type == 'svm':
        clf = SVC(kernel='rbf', probability=True)
    elif model_type == 'rf':
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unknown model: {model_type}")
    cv_obj = LeaveOneOut() if cv == 'loo' else int(cv)
    scores = cross_val_score(clf, X, y, cv=cv_obj, scoring='accuracy')
    clf.fit(X, y)
    result = {
        'model': clf, 'cv_accuracy_mean': scores.mean(),
        'cv_accuracy_std': scores.std(), 'cv_scores': scores,
    }
    if model_type == 'rf':
        result['feature_importance'] = clf.feature_importances_
    return result


def compute_replicate_correlation(df, cols):
    """Compute Pearson correlation between replicates."""
    if len(cols) < 2:
        return None
    x = np.log2(df[cols[0]].replace(0, np.nan))
    y = np.log2(df[cols[1]].replace(0, np.nan))
    valid = x.notna() & y.notna()
    if valid.sum() < 3:
        return None
    r, p = stats.pearsonr(x[valid], y[valid])
    return {'r': r, 'p': p, 'n': valid.sum()}
