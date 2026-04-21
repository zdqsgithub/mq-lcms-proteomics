"""
MaxQuant LC-MS/MS Proteomics - Visualization Module
====================================================
Publication-quality figures for proteomics analysis.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Global style
STYLE = {
    'figure.dpi': 150, 'savefig.dpi': 150,
    'font.family': 'sans-serif', 'font.size': 10,
    'axes.titlesize': 13, 'axes.labelsize': 11,
    'figure.facecolor': 'white',
}
plt.rcParams.update(STYLE)
sns.set_style("whitegrid")


def plot_msms_summary(qc_df, groups, colors, output_dir):
    """Bar chart of MS/MS submitted vs identified per sample."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(qc_df))
    
    bar_colors = []
    for s in qc_df['sample']:
        matched = '#999999'
        for g, samps in groups.items():
            if any(sm in str(s) for sm in samps):
                matched = colors[g]; break
        bar_colors.append(matched)
    
    ax = axes[0]
    ax.bar(x, qc_df['msms_submitted'], color=[c+'55' for c in bar_colors], edgecolor=bar_colors)
    ax.bar(x, qc_df['msms_identified'], color=bar_colors)
    ax.set_xticks(x)
    ax.set_xticklabels(qc_df['sample'], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('MS/MS Spectra'); ax.set_title('MS/MS Identification')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v/1e3:.0f}k'))
    
    ax = axes[1]
    ax.bar(x, qc_df['peptides'], color=bar_colors)
    ax.set_xticks(x)
    ax.set_xticklabels(qc_df['sample'], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Unique Peptide Sequences'); ax.set_title('Peptide Identification')
    
    plt.tight_layout()
    path = Path(output_dir) / 'fig01_msms_summary.png'
    fig.savefig(path, bbox_inches='tight'); plt.close(fig)
    return path


def plot_protein_counts(pg, quant_cols, colors, output_dir):
    """Bar chart of protein groups detected per group."""
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = {}
    for g, cols in quant_cols.items():
        counts[g] = (pg[cols] > 0).any(axis=1).sum()
    bars = ax.bar(counts.keys(), counts.values(),
                  color=[colors[g] for g in counts], width=0.5)
    for bar, v in zip(bars, counts.values()):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                str(v), ha='center', fontweight='bold')
    ax.set_ylabel('Protein Groups'); ax.set_title('Proteins Detected per Group')
    sns.despine(); plt.tight_layout()
    path = Path(output_dir) / 'fig02_proteins_per_group.png'
    fig.savefig(path, bbox_inches='tight'); plt.close(fig)
    return path


def plot_missing_values(pg, all_cols, output_dir):
    """Heatmap of missing values across samples."""
    missing = ((pg[all_cols] == 0) | pg[all_cols].isna()).astype(int)
    short = [c.split(' ', 1)[-1] if ' ' in c else c for c in all_cols]
    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(missing.T, cmap=['#2d6a4f','#d62828'],
                yticklabels=short, xticklabels=False, ax=ax,
                cbar_kws={'label': 'Missing', 'ticks': [0,1]})
    ax.set_title(f'Missing Values — {len(pg)} Protein Groups')
    plt.tight_layout()
    path = Path(output_dir) / 'fig03_missing_values.png'
    fig.savefig(path, bbox_inches='tight'); plt.close(fig)
    return path


def plot_intensity_distribution(pg, quant_cols, colors, output_dir):
    """Histogram of log2 intensity distribution."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for g, cols in quant_cols.items():
        for c in cols:
            vals = np.log2(pg[c].replace(0, np.nan).dropna())
            ax.hist(vals, bins=50, alpha=0.4, color=colors[g], density=True)
    handles = [plt.Line2D([0],[0], color=colors[g], lw=4, label=g) for g in quant_cols]
    ax.legend(handles=handles, prop={'weight':'bold'})
    ax.set_xlabel('log₂(intensity)'); ax.set_ylabel('Density')
    ax.set_title('Intensity Distribution')
    sns.despine(); plt.tight_layout()
    path = Path(output_dir) / 'fig04_intensity_distribution.png'
    fig.savefig(path, bbox_inches='tight'); plt.close(fig)
    return path


def plot_replicate_correlation(pg, quant_cols, colors, output_dir):
    """Scatter plot of replicate correlations."""
    from scipy import stats as sp_stats
    n_groups = len(quant_cols)
    fig, axes = plt.subplots(1, n_groups, figsize=(5*n_groups, 5))
    if n_groups == 1: axes = [axes]
    for i, (g, cols) in enumerate(quant_cols.items()):
        ax = axes[i]
        if len(cols) < 2:
            ax.text(0.5, 0.5, 'Single replicate', ha='center', transform=ax.transAxes)
            continue
        x = np.log2(pg[cols[0]].replace(0, np.nan))
        y = np.log2(pg[cols[1]].replace(0, np.nan))
        valid = x.notna() & y.notna()
        if valid.sum() > 2:
            r, _ = sp_stats.pearsonr(x[valid], y[valid])
            ax.scatter(x[valid], y[valid], s=8, alpha=0.4, color=colors[g])
            ax.set_title(f'{g} (r={r:.4f})')
            lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                    max(ax.get_xlim()[1], ax.get_ylim()[1])]
            ax.plot(lims, lims, 'k--', alpha=0.3)
        ax.set_xlabel('log₂ Rep 1'); ax.set_ylabel('log₂ Rep 2')
    sns.despine(); plt.tight_layout()
    path = Path(output_dir) / 'fig05_replicate_correlation.png'
    fig.savefig(path, bbox_inches='tight'); plt.close(fig)
    return path


def plot_venn(pg, quant_cols, colors, output_dir):
    """Venn diagram of protein overlap between groups."""
    try:
        from matplotlib_venn import venn2, venn3
    except ImportError:
        return None
    
    sets = {}
    for g, cols in quant_cols.items():
        sets[g] = set(pg[(pg[cols] > 0).any(axis=1)].index)
    
    fig, ax = plt.subplots(figsize=(7, 7))
    names = list(sets.keys())
    if len(names) == 2:
        venn2([sets[names[0]], sets[names[1]]], set_labels=names,
              set_colors=(colors[names[0]], colors[names[1]]), alpha=0.6, ax=ax)
    elif len(names) >= 3:
        venn3([sets[names[0]], sets[names[1]], sets[names[2]]],
              set_labels=names[:3],
              set_colors=(colors[names[0]], colors[names[1]], colors[names[2]]),
              alpha=0.6, ax=ax)
    ax.set_title('Protein Group Overlap')
    plt.tight_layout()
    path = Path(output_dir) / 'fig06_venn_diagram.png'
    fig.savefig(path, bbox_inches='tight'); plt.close(fig)
    return path


def plot_volcano(diff_results, ga, gb, colors, fc_thresh, pval_thresh, output_dir, idx=7):
    """Volcano plot for one comparison."""
    fig, ax = plt.subplots(figsize=(8, 6))
    df = diff_results.copy()
    sig_up = df[(df['log2FC'] > fc_thresh) & (df['pvalue'] < pval_thresh)]
    sig_dn = df[(df['log2FC'] < -fc_thresh) & (df['pvalue'] < pval_thresh)]
    ns = df[~((abs(df['log2FC']) > fc_thresh) & (df['pvalue'] < pval_thresh))]
    
    ax.scatter(ns['log2FC'], ns['-log10p'], s=8, alpha=0.3, c='#bbbbbb', label='NS')
    ax.scatter(sig_up['log2FC'], sig_up['-log10p'], s=12, alpha=0.7,
               c=colors.get(ga, '#E15759'), label=f'Up {ga} ({len(sig_up)})')
    ax.scatter(sig_dn['log2FC'], sig_dn['-log10p'], s=12, alpha=0.7,
               c=colors.get(gb, '#4E79A7'), label=f'Up {gb} ({len(sig_dn)})')
    ax.axhline(-np.log10(pval_thresh), ls='--', color='grey', lw=0.8)
    ax.axvline(fc_thresh, ls='--', color='grey', lw=0.8)
    ax.axvline(-fc_thresh, ls='--', color='grey', lw=0.8)
    ax.set_xlabel(f'log₂FC ({ga}/{gb})'); ax.set_ylabel('-log₁₀(p)')
    ax.set_title(f'{ga} vs {gb}')
    ax.legend(fontsize=8, prop={'weight':'bold'})
    sns.despine(); plt.tight_layout()
    path = Path(output_dir) / f'fig{idx:02d}_volcano_{ga}_vs_{gb}.png'
    fig.savefig(path, bbox_inches='tight'); plt.close(fig)
    return path


def plot_allergen_heatmap(allergens, all_cols, output_dir):
    """Clustered heatmap of allergen-related proteins."""
    heat = allergens[all_cols].replace(0, np.nan).apply(np.log2)
    heat.columns = [c.split(' ', 1)[-1] if ' ' in c else c for c in all_cols]
    
    fig, ax = plt.subplots(figsize=(12, max(9, len(heat)*0.42)))
    sns.heatmap(heat, cmap='YlOrRd', ax=ax, linewidths=0.3,
                cbar_kws={'label': 'log₂(intensity)'})
    ax.set_title('Allergen-Related Proteins', fontsize=14, fontweight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10, fontweight='bold')
    plt.tight_layout()
    path = Path(output_dir) / 'fig10_allergen_heatmap.png'
    fig.savefig(path, bbox_inches='tight'); plt.close(fig)
    return path


def plot_pca(coords, var_ratio, labels, colors_map, groups, output_dir):
    """PCA scatter plot colored by group."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for g, samps in groups.items():
        idx = [i for i, l in enumerate(labels) if any(s in l for s in samps)]
        if idx:
            ax.scatter(coords[idx, 0], coords[idx, 1], s=80,
                      color=colors_map.get(g, '#999'), label=g, edgecolors='white')
            for i in idx:
                ax.annotate(labels[i].split('-')[-1] if '-' in labels[i] else labels[i],
                           (coords[i,0], coords[i,1]), fontsize=7, alpha=0.7)
    ax.set_xlabel(f'PC1 ({var_ratio[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({var_ratio[1]*100:.1f}%)')
    ax.set_title('PCA of Protein Quantification')
    ax.legend(prop={'weight':'bold'})
    sns.despine(); plt.tight_layout()
    path = Path(output_dir) / 'fig12_pca.png'
    fig.savefig(path, bbox_inches='tight'); plt.close(fig)
    return path


def plot_top_proteins(pg, quant_cols, colors, n=20, output_dir='.'):
    """Horizontal bar chart of top N most abundant proteins."""
    all_cols = [c for cols in quant_cols.values() for c in cols]
    pg_copy = pg.copy()
    pg_copy['avg_quant'] = pg_copy[all_cols].replace(0, np.nan).mean(axis=1)
    top = pg_copy.nlargest(n, 'avg_quant')
    top['label'] = top['description'].where(
        top['description'] != '', top.get('Majority protein IDs', 'Unknown')
    ).apply(lambda x: str(x)[:55])
    
    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(top))
    bw = 0.8 / len(quant_cols)
    for i, (g, cols) in enumerate(quant_cols.items()):
        vals = top[cols].replace(0, np.nan).mean(axis=1).fillna(0) / 1e6
        ax.barh(y_pos + i*bw, vals, bw, color=colors[g], label=g, alpha=0.85)
    ax.set_yticks(y_pos + bw); ax.set_yticklabels(top['label'].values, fontsize=8)
    ax.set_xlabel('Average Intensity (×10⁶)'); ax.set_title(f'Top {n} Most Abundant Proteins')
    ax.legend(prop={'weight':'bold'}); ax.invert_yaxis()
    sns.despine(); plt.tight_layout()
    path = Path(output_dir) / f'fig09_top{n}_proteins.png'
    fig.savefig(path, bbox_inches='tight'); plt.close(fig)
    return path
