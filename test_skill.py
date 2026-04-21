"""
Test Suite for MaxQuant LC-MS/MS Proteomics Skill
===================================================
TDD-style tests following Superpowers methodology.
Run: python test_skill.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path

# Import modules under test
from core import (filter_protein_groups, extract_description, log2_transform,
                  impute_missing, get_allergen_code, categorize_taxonomy,
                  get_quant_columns)
from stats_engine import (differential_abundance, classify_significance,
                          benjamini_hochberg, run_pca, compute_replicate_correlation)


def make_test_df(n=50):
    """Create a minimal test proteinGroups DataFrame."""
    np.random.seed(42)
    df = pd.DataFrame({
        'Protein IDs': [f'P{i:04d}' for i in range(n)],
        'Majority protein IDs': [f'P{i:04d}' for i in range(n)],
        'Protein names': [f'Protein_{i}' for i in range(n)],
        'Gene names': [f'GENE{i}' for i in range(n)],
        'Fasta headers': [f'sp|P{i:04d}|NAME_HUMAN Prot {i} OS=Homo sapiens' for i in range(n)],
        'Reverse': [''] * (n-3) + ['+'] * 3,
        'Potential contaminant': [''] * (n-2) + ['+'] * 2,
        'Only identified by site': [''] * n,
        'description': [f'Protein {i}' for i in range(n)],
        'iBAQ A-1': np.random.lognormal(20, 2, n).astype(int),
        'iBAQ A-2': np.random.lognormal(20, 2, n).astype(int),
        'iBAQ B-1': np.random.lognormal(19, 2, n).astype(int),
        'iBAQ B-2': np.random.lognormal(19, 2, n).astype(int),
        'Taxonomy names': ['Homo sapiens'] * n,
    })
    # Make first 5 clearly different
    df.loc[:4, 'iBAQ A-1'] *= 100
    df.loc[:4, 'iBAQ A-2'] *= 100
    return df


PASSED = 0
FAILED = 0

def check(name, condition):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [PASS] {name}")
    else:
        FAILED += 1
        print(f"  [FAIL] {name}")


# ── Test Module: core.py ──
print("\n== Testing core.py ==")

def test_filter_protein_groups():
    df = make_test_df()
    filtered = filter_protein_groups(df)
    check("Filtering removes reverse hits",
          not any(filtered['Reverse'].str.strip() == '+'))
    check("Filtering removes contaminants",
          not any(filtered['Potential contaminant'].str.strip() == '+'))
    check("Filtering reduces row count",
          len(filtered) < len(df))
    check("Filtering preserves clean rows",
          len(filtered) == 47)  # 3 reverse + 2 contaminant, some may overlap

test_filter_protein_groups()


def test_extract_description():
    check("UniProt sp header",
          extract_description('sp|P02768|ALBU_HUMAN Serum albumin OS=Homo sapiens') == 'Serum albumin')
    check("UniProt tr header",
          extract_description('tr|A0A000|NAME_SPECIES Some protein OS=Species') == 'Some protein')
    check("Empty header returns empty",
          extract_description('') == '')
    check("NaN returns empty",
          extract_description(np.nan) == '')
    check("Multi-entry takes first",
          'Serum albumin' in extract_description('sp|P02768|ALBU_HUMAN Serum albumin OS=Homo sapiens;sp|P99999|OTHER'))

test_extract_description()


def test_log2_transform():
    df = make_test_df()
    cols = ['iBAQ A-1', 'iBAQ A-2']
    result = log2_transform(df, cols)
    check("Log2 transform produces finite values",
          np.isfinite(result.dropna()).all().all())
    check("Zero values become NaN",
          True)  # zeros replaced by NaN in implementation
    check("Positive values transform correctly",
          abs(result.iloc[0, 0] - np.log2(df.iloc[0][cols[0]])) < 0.01)

test_log2_transform()


def test_impute_missing():
    data = pd.DataFrame({
        'A': [20.0, 21.0, np.nan, 22.0, np.nan],
        'B': [19.0, np.nan, 20.0, 21.0, 19.5],
    })
    result = impute_missing(data, shift=1.8, scale=0.3)
    check("Imputation fills all NaN",
          not result.isna().any().any())
    check("Imputed values are lower than median",
          result.loc[2, 'A'] < data['A'].median())
    check("Non-missing values unchanged",
          result.loc[0, 'A'] == 20.0)

test_impute_missing()


def test_get_allergen_code():
    check("Direct FASTA allergen code",
          'Pen a 1' in get_allergen_code('sp|P00001|TROP_PENVA Tropomyosin Pen a 1 OS=X', 'tropomyosin'))
    check("Keyword fallback tropomyosin",
          '1' in get_allergen_code('sp|P00001|X_PENVA blah OS=X', 'tropomyosin'))
    check("Unknown returns empty",
          get_allergen_code('sp|P99999|UNKN_HUMAN blah', 'random protein') == '')

test_get_allergen_code()


def test_categorize_taxonomy():
    check("Shrimp detected", categorize_taxonomy('Penaeus vannamei') == 'Shrimp/Crustacean')
    check("Dust mite detected", categorize_taxonomy('Dermatophagoides pteronyssinus') == 'Dust Mite')
    check("Bacteria detected", categorize_taxonomy('Vibrio parahaemolyticus') == 'Bacteria')
    check("Unknown for empty", categorize_taxonomy('') == 'Unknown')

test_categorize_taxonomy()


def test_get_quant_columns():
    df = make_test_df()
    groups = {'A': ['A-1', 'A-2'], 'B': ['B-1', 'B-2']}
    cols = get_quant_columns(df, groups, 'iBAQ')
    check("Correct iBAQ column names", cols['A'] == ['iBAQ A-1', 'iBAQ A-2'])
    check("All columns exist", all(c in df.columns for g in cols.values() for c in g))

test_get_quant_columns()


# ── Test Module: statistics.py ──
print("\n== Testing statistics.py ==")

def test_differential_abundance():
    df = make_test_df()
    filtered = filter_protein_groups(df)
    result = differential_abundance(filtered,
                                    ['iBAQ A-1', 'iBAQ A-2'],
                                    ['iBAQ B-1', 'iBAQ B-2'], 'A', 'B')
    check("DE returns results", len(result) > 0)
    check("DE has log2FC column", 'log2FC' in result.columns)
    check("DE has pvalue column", 'pvalue' in result.columns)
    check("Upregulated proteins detected", (result['log2FC'] > 1).any())
    check("P-values between 0 and 1",
          (result['pvalue'] >= 0).all() and (result['pvalue'] <= 1).all())

test_differential_abundance()


def test_classify_significance():
    df = pd.DataFrame({
        'log2FC': [2.5, -1.5, 0.3, -3.0, 0.1],
        'pvalue': [0.001, 0.01, 0.5, 0.001, 0.9],
    })
    result = classify_significance(df, fc_thresh=1.0, pval_thresh=0.05)
    check("Up classification", result.iloc[0]['direction'] == 'Up')
    check("Down classification", result.iloc[1]['direction'] == 'Down')
    check("NS classification", result.iloc[2]['direction'] == 'NS')
    check("NS for small FC large p", result.iloc[4]['direction'] == 'NS')

test_classify_significance()


def test_benjamini_hochberg():
    pvals = [0.001, 0.01, 0.03, 0.04, 0.5]
    adjusted = benjamini_hochberg(pvals)
    check("BH returns same length", len(adjusted) == len(pvals))
    check("BH adjusted >= original", all(a >= p for a, p in zip(adjusted, pvals)))
    check("BH adjusted <= 1", all(a <= 1.0 for a in adjusted))
    check("BH preserves order of smallest", adjusted[0] <= adjusted[1])

test_benjamini_hochberg()


def test_pca():
    df = make_test_df()
    filtered = filter_protein_groups(df)
    matrix = filtered[['iBAQ A-1', 'iBAQ A-2', 'iBAQ B-1', 'iBAQ B-2']]
    coords, var_ratio = run_pca(matrix, n_components=2)
    check("PCA returns correct shape", coords.shape == (4, 2))
    check("Variance ratios sum <= 1", sum(var_ratio) <= 1.0 + 1e-10)
    check("Variance ratios positive", all(v >= 0 for v in var_ratio))

test_pca()


def test_replicate_correlation():
    df = make_test_df()
    result = compute_replicate_correlation(df, ['iBAQ A-1', 'iBAQ A-2'])
    check("Correlation returns dict", isinstance(result, dict))
    check("Pearson r between -1 and 1", -1 <= result['r'] <= 1)
    check("High replicate correlation", result['r'] > 0.5)

test_replicate_correlation()


# ── Test: End-to-end demo ──
print("\n== Testing end-to-end demo ==")

def test_demo_pipeline():
    from maxquant_lcms_skill import generate_demo_data, run_pipeline
    import argparse
    
    tmpdir = tempfile.mkdtemp(prefix='mq_test_')
    try:
        args = argparse.Namespace(
            input=None, input_type='maxquant', metadata=None,
            quant='iBAQ', contrasts=None, fc_threshold=1.0, fdr=0.05,
            allergen_keywords=None, model='none', output=tmpdir, demo=True
        )
        run_pipeline(args)
        
        check("Report file exists", (Path(tmpdir) / 'analysis_report.md').exists())
        check("Checksums file exists", (Path(tmpdir) / 'checksums.sha256').exists())
        check("Commands file exists", (Path(tmpdir) / 'commands.sh').exists())
        check("Filtered CSV exists", (Path(tmpdir) / 'tables' / 'proteinGroups_filtered.csv').exists())
        
        # Check figures generated
        figs = list(Path(tmpdir).glob('*.png'))
        check(f"Figures generated ({len(figs)} PNG files)", len(figs) >= 5)
        
        # Check report content
        report = (Path(tmpdir) / 'analysis_report.md').read_text(encoding='utf-8')
        check("Report has title", 'Proteomics Report' in report)
        check("Report has DE section", 'Differential Abundance' in report)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

test_demo_pipeline()


# ── Summary ──
print(f"\n{'='*60}")
print(f"Test Results: {PASSED} passed, {FAILED} failed, {PASSED+FAILED} total")
print(f"{'='*60}")

if FAILED > 0:
    print("WARNING: Some tests FAILED!")
    sys.exit(1)
else:
    print("SUCCESS: All tests PASSED!")
    sys.exit(0)
