"""
MaxQuant LC-MS/MS Proteomics - Statistical Analysis Module
==========================================================
Differential abundance, multiple testing correction, modeling.
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')


def differential_abundance(df, cols_a, cols_b, label_a='A', label_b='B'):
    """Compute log2FC and p-value between two groups using Welch's t-test."""
    results = []
    for idx, row in df.iterrows():
        a = row[cols_a].replace(0, np.nan).dropna().values.astype(float)
        b = row[cols_b].replace(0, np.nan).dropna().values.astype(float)
        if len(a) < 1 or len(b) < 1:
            continue
        mean_a = np.mean(np.log2(a))
        mean_b = np.mean(np.log2(b))
        log2fc = mean_a - mean_b
        if len(a) >= 2 and len(b) >= 2:
            _, pval = stats.ttest_ind(np.log2(a), np.log2(b), equal_var=False)
        else:
            pval = 1.0
        results.append({
            'index': idx,
            'Protein IDs': row.get('Protein IDs', ''),
            'Protein names': row.get('Protein names', ''),
            'description': row.get('description', ''),
            'Gene names': row.get('Gene names', ''),
            'Majority protein IDs': row.get('Majority protein IDs', ''),
            'log2FC': log2fc,
            'pvalue': pval,
            '-log10p': -np.log10(max(pval, 1e-300)),
        })
    return pd.DataFrame(results)


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
    t_stat = stats.t.ppf(1 - fdr / 2, df=4)  # default df=4
    fc_abs = np.abs(log2fc)
    neg_log10p = -np.log10(np.maximum(pvalue, 1e-300))
    # s0-adjusted threshold curve
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


def run_pca(data_matrix, n_components=2):
    """Run PCA on protein quantification matrix."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data_matrix.fillna(0).T)
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(scaled)
    return coords, pca.explained_variance_ratio_


def train_classifier(X, y, model_type='svm', cv='loo'):
    """Train a classifier for sample group prediction."""
    if model_type == 'svm':
        clf = SVC(kernel='rbf', probability=True)
    elif model_type == 'rf':
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unknown model: {model_type}")
    
    if cv == 'loo':
        cv_obj = LeaveOneOut()
    else:
        cv_obj = int(cv)
    
    scores = cross_val_score(clf, X, y, cv=cv_obj, scoring='accuracy')
    clf.fit(X, y)
    
    result = {
        'model': clf,
        'cv_accuracy_mean': scores.mean(),
        'cv_accuracy_std': scores.std(),
        'cv_scores': scores,
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
