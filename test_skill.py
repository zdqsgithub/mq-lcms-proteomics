"""
Test Suite v2 for MaxQuant LC-MS/MS Proteomics Skill
=====================================================
Covers: core v2 (external DBs), vectorized stats, time-course, modes.
Run: python test_skill.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path

from core import (filter_protein_groups, extract_description, log2_transform,
                  impute_missing, get_allergen_code, categorize_taxonomy,
                  get_quant_columns, auto_detect_groups)
from stats_engine import (differential_abundance, classify_significance,
                          benjamini_hochberg, run_pca, compute_replicate_correlation,
                          timecourse_analysis)

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


def make_test_df(n=50):
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
    df.loc[:4, 'iBAQ A-1'] *= 100
    df.loc[:4, 'iBAQ A-2'] *= 100
    return df


# ══ core.py tests ══
print("\n== Testing core.py v2 ==")

def test_filter():
    df = make_test_df()
    filtered = filter_protein_groups(df)
    check("Filter removes reverse", not any(filtered['Reverse'].str.strip() == '+'))
    check("Filter removes contaminants", not any(filtered['Potential contaminant'].str.strip() == '+'))
    check("Filter reduces count", len(filtered) < len(df))
    check("Filter count correct", len(filtered) == 47)
test_filter()

def test_extract_description():
    check("sp header", extract_description('sp|P02768|ALBU_HUMAN Serum albumin OS=Homo sapiens') == 'Serum albumin')
    check("tr header", extract_description('tr|A0A000|NAME_SPECIES Some protein OS=Species') == 'Some protein')
    check("Empty", extract_description('') == '')
    check("NaN", extract_description(np.nan) == '')
    check("Multi-entry", 'Serum albumin' in extract_description('sp|P02768|ALBU_HUMAN Serum albumin OS=Homo sapiens;sp|P99999|OTHER'))
test_extract_description()

def test_log2():
    df = make_test_df()
    result = log2_transform(df, ['iBAQ A-1', 'iBAQ A-2'])
    check("Log2 finite", np.isfinite(result.dropna()).all().all())
    check("Log2 correct", abs(result.iloc[0, 0] - np.log2(df.iloc[0]['iBAQ A-1'])) < 0.01)
test_log2()

def test_impute():
    data = pd.DataFrame({'A': [20.0, 21.0, np.nan, 22.0, np.nan], 'B': [19.0, np.nan, 20.0, 21.0, 19.5]})
    result = impute_missing(data)
    check("Impute fills NaN", not result.isna().any().any())
    check("Imputed < median", result.loc[2, 'A'] < data['A'].median())
    check("Non-missing preserved", result.loc[0, 'A'] == 20.0)
test_impute()

def test_allergen_v2():
    # Crustacean allergens
    check("Shrimp tropomyosin", '1' in get_allergen_code('sp|P00001|TROP_PENVA Tropomyosin Pen a 1 OS=X', 'tropomyosin'))
    # Plant allergens (v2 new)
    check("Profilin mapped", '4' in get_allergen_code('sp|P00001|PROF_ARTVU blah OS=X', 'profilin'))
    check("Polcalcin mapped", '5' in get_allergen_code('sp|P00001|POLC_ARTVU blah OS=X', 'polcalcin'))
    check("nsLTP mapped", '3' in get_allergen_code('sp|P00001|LTP_ARTVU blah OS=X', 'non-specific lipid-transfer protein'))
    check("CPI mapped", 'CPI' in get_allergen_code('sp|P00001|CPI_ARTVU blah OS=X', 'cysteine proteinase inhibitor'))
    check("Unknown empty", get_allergen_code('sp|P99999|UNKN_HUMAN blah', 'random protein') == '')
test_allergen_v2()

def test_taxonomy_v2():
    check("Shrimp", categorize_taxonomy('Penaeus vannamei') == 'Shrimp/Crustacean')
    check("Mite", categorize_taxonomy('Dermatophagoides pteronyssinus') == 'Dust Mite')
    check("Bacteria", categorize_taxonomy('Vibrio parahaemolyticus') == 'Bacteria')
    # v2 new plant taxa
    check("Mugwort", categorize_taxonomy('Artemisia vulgaris') == 'Mugwort/Artemisia')
    check("Ragweed", categorize_taxonomy('Ambrosia artemisiifolia') == 'Ragweed')
    check("Birch", categorize_taxonomy('Betula pendula') == 'Birch')
    check("Grass", categorize_taxonomy('Lolium perenne') == 'Grass Pollen')
    check("Unknown empty", categorize_taxonomy('') == 'Unknown')
test_taxonomy_v2()

def test_auto_detect():
    df = make_test_df()
    groups, colors = auto_detect_groups(df, 'iBAQ')
    check("Auto-detect finds groups", len(groups) >= 2)
    check("Auto-detect has colors", len(colors) == len(groups))
test_auto_detect()

def test_quant_cols():
    df = make_test_df()
    groups = {'A': ['A-1', 'A-2'], 'B': ['B-1', 'B-2']}
    cols = get_quant_columns(df, groups, 'iBAQ')
    check("Correct col names", cols['A'] == ['iBAQ A-1', 'iBAQ A-2'])
    check("Cols exist", all(c in df.columns for g in cols.values() for c in g))
test_quant_cols()


# ══ stats_engine.py tests ══
print("\n== Testing stats_engine.py v2 ==")

def test_de_vectorized():
    df = make_test_df()
    filtered = filter_protein_groups(df)
    result = differential_abundance(filtered, ['iBAQ A-1', 'iBAQ A-2'], ['iBAQ B-1', 'iBAQ B-2'])
    check("DE returns results", len(result) > 0)
    check("DE has log2FC", 'log2FC' in result.columns)
    check("DE has pvalue", 'pvalue' in result.columns)
    check("DE detects up", (result['log2FC'] > 1).any())
    check("P-values valid", (result['pvalue'] >= 0).all() and (result['pvalue'] <= 1).all())
test_de_vectorized()

def test_timecourse():
    np.random.seed(42)
    n = 20
    df = pd.DataFrame({
        'iBAQ D0-1': np.random.lognormal(20, 1, n).astype(int),
        'iBAQ D0-2': np.random.lognormal(20, 1, n).astype(int),
        'iBAQ D3-1': np.random.lognormal(19, 1, n).astype(int),
        'iBAQ D3-2': np.random.lognormal(19, 1, n).astype(int),
        'iBAQ D7-1': np.random.lognormal(18, 1, n).astype(int),
        'iBAQ D7-2': np.random.lognormal(18, 1, n).astype(int),
        'description': [f'Protein {i}' for i in range(n)],
        'label': [f'Protein {i}' for i in range(n)],
    })
    groups = {'D0': ['iBAQ D0-1','iBAQ D0-2'], 'D3': ['iBAQ D3-1','iBAQ D3-2'], 'D7': ['iBAQ D7-1','iBAQ D7-2']}
    result = timecourse_analysis(df, groups, ['D0','D3','D7'])
    check("TC has mean cols", 'mean_D0' in result.columns)
    check("TC has pct cols", 'pct_D7' in result.columns)
    check("TC has FC cols", 'log2FC_D7' in result.columns)
    check("TC has trend", 'trend' in result.columns)
    check("TC baseline=100", (result['pct_D0'] == 100).all())
    check("TC has trends", set(result['trend'].unique()).issubset({'Degrading','Stable','Increasing'}))
test_timecourse()

def test_classify():
    df = pd.DataFrame({'log2FC': [2.5, -1.5, 0.3, -3.0], 'pvalue': [0.001, 0.01, 0.5, 0.001]})
    result = classify_significance(df)
    check("Up", result.iloc[0]['direction'] == 'Up')
    check("Down", result.iloc[1]['direction'] == 'Down')
    check("NS", result.iloc[2]['direction'] == 'NS')
test_classify()

def test_bh():
    pvals = [0.001, 0.01, 0.03, 0.04, 0.5]
    adj = benjamini_hochberg(pvals)
    check("BH same length", len(adj) == len(pvals))
    check("BH adj >= orig", all(a >= p for a, p in zip(adj, pvals)))
    check("BH adj <= 1", all(a <= 1.0 for a in adj))
test_bh()

def test_pca():
    df = make_test_df()
    filtered = filter_protein_groups(df)
    matrix = filtered[['iBAQ A-1', 'iBAQ A-2', 'iBAQ B-1', 'iBAQ B-2']]
    coords, var = run_pca(matrix)
    check("PCA shape", coords.shape == (4, 2))
    check("PCA variance", sum(var) <= 1.0 + 1e-10)
test_pca()

def test_correlation():
    df = make_test_df()
    r = compute_replicate_correlation(df, ['iBAQ A-1', 'iBAQ A-2'])
    check("Corr returns dict", isinstance(r, dict))
    check("Corr r range", -1 <= r['r'] <= 1)
test_correlation()


# ══ End-to-end tests ══
print("\n== Testing end-to-end ==")

def test_demo_comparison():
    from maxquant_lcms_skill import run_comparison
    import argparse
    tmpdir = tempfile.mkdtemp(prefix='mq_v2_comp_')
    try:
        args = argparse.Namespace(
            input=None, input_dir=None, input_type='maxquant', metadata=None, quant='iBAQ',
            mode='comparison', contrasts=None, fc_threshold=1.0, fdr=0.05,
            allergen_keywords=None, model='none', output=tmpdir, demo=True)
        run_comparison(args)
        check("Comparison report exists", (Path(tmpdir) / 'analysis_report.md').exists())
        check("Comparison checksums", (Path(tmpdir) / 'checksums.sha256').exists())
        check("Comparison figures", len(list(Path(tmpdir).glob('*.png'))) >= 5)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
test_demo_comparison()

def test_demo_stability():
    from maxquant_lcms_skill import run_stability
    import argparse
    tmpdir = tempfile.mkdtemp(prefix='mq_v2_stab_')
    try:
        args = argparse.Namespace(
            input=None, input_dir=None, input_type='maxquant', metadata=None, quant='iBAQ',
            mode='stability', contrasts=None, fc_threshold=1.0, fdr=0.05,
            allergen_keywords=None, model='none', output=tmpdir, demo=True)
        run_stability(args)
        check("Stability report exists", (Path(tmpdir) / 'stability_report.md').exists())
        check("Stability figures", len(list(Path(tmpdir).glob('*.png'))) >= 3)
        check("Stability CSV", (Path(tmpdir) / 'tables' / 'stability_summary.csv').exists())
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
test_demo_stability()


# ══ degradation_routes.py tests ══
print("\n== Testing degradation_routes.py ==")

from degradation_routes import (assign_functional_category, functional_enrichment,
                                detect_semi_tryptic, semi_tryptic_kinetics,
                                inventory_proteases_phosphatases,
                                count_deamidation_motifs, peptide_appearance)

def test_functional_category():
    check("Cat: HSP", assign_functional_category("Heat shock protein 70") == 'Chaperone/HSP')
    check("Cat: GST", assign_functional_category("Glutathione S-transferase") == 'Redox/Antioxidant')
    check("Cat: tropomyosin", assign_functional_category("Tropomyosin") == 'Structural/Muscle')
    check("Cat: calpain", assign_functional_category("Calpain clp1-like") == 'Other')  # calpain is protease, not in func cats
    check("Cat: unknown", assign_functional_category("Unknown protein XYZ") == 'Other')
test_functional_category()

def test_functional_enrichment():
    df = pd.DataFrame({
        'description': ['Heat shock protein 70', 'Glutathione transferase', 'Tropomyosin',
                        'Myosin heavy chain', 'Unknown protein', 'Enolase'],
        'trend': ['Degrading', 'Degrading', 'Increasing', 'Increasing', 'Stable', 'Stable']
    })
    cat_counts, cat_pcts, tg = functional_enrichment(df)
    check("Enrich: counts OK", 'Degrading' in cat_counts.columns)
    check("Enrich: pcts OK", cat_pcts['Degrading'].sum() > 0)
    check("Enrich: groups OK", len(tg['Degrading']) == 2)
test_functional_enrichment()

def test_semi_tryptic():
    df = pd.DataFrame({
        'Sequence': ['PEPTIDEK', 'SAMPLER', 'NOTRYPTIC'],
        'Amino acid before': ['R', 'K', 'A'],
        'Amino acid after': ['L', '-', 'G'],
    })
    result = detect_semi_tryptic(df)
    check("Semi: fully tryptic", result.iloc[0]['cleavage_type'] == 'Fully tryptic')
    check("Semi: c-term only", result.iloc[1]['cleavage_type'] == 'Fully tryptic')  # R end + K before
    check("Semi: non-tryptic detected", result.iloc[2]['cleavage_type'] in ['Semi-tryptic', 'Non-tryptic'])
test_semi_tryptic()

def test_protease_inventory():
    df = pd.DataFrame({
        'description': ['Calpain clp1-like protein', 'Heat shock protein 70',
                        'Serine/threonine-protein phosphatase'],
        'trend': ['Increasing', 'Degrading', 'Degrading']
    })
    prot, phos = inventory_proteases_phosphatases(df)
    check("Inv: calpain found", len(prot) >= 1)
    check("Inv: calpain high risk", prot.iloc[0]['risk'] == 'HIGH')
    check("Inv: phosphatase found", len(phos) >= 1)
test_protease_inventory()

def test_deamidation_motifs():
    df = pd.DataFrame({'Sequence': ['ANGSTROM', 'PEPTIDE', 'MINSTER']})
    count = count_deamidation_motifs(df)
    check("Deamid: NG found", count >= 1)  # ANGSTROM has NG
test_deamidation_motifs()

def test_peptide_appearance():
    df = pd.DataFrame({
        'Intensity D0': [100, 200, 0, 300],
        'Intensity D7': [0, 200, 300, 400],
    })
    groups = {'D0': ['D0'], 'D7': ['D7']}
    result = peptide_appearance(df, groups)
    check("Pep appear: lost count", result['lost_from_baseline'] >= 1)
    check("Pep appear: gained count", result['gained_at_last'] >= 1)
test_peptide_appearance()


# ══ v2.2 new function tests ══
print("\n== Testing v2.2 functions ==")

from degradation_routes import (coverage_kinetics, peptide_gravy,
                                sequence_composition, analyze_deamidation_sites)

def test_gravy():
    check("GRAVY: hydrophobic", peptide_gravy('AVILM') > 0)
    check("GRAVY: hydrophilic", peptide_gravy('DERK') < 0)
    check("GRAVY: returns float", isinstance(peptide_gravy('PEPTIDE'), float))
test_gravy()

def test_coverage_kinetics():
    df = pd.DataFrame({
        'Proteins': ['P001', 'P001', 'P001', 'P002', 'P002'],
        'Sequence': ['AAAK', 'BBBK', 'CCCK', 'DDDK', 'EEEK'],
        'Intensity D0': [100, 200, 0, 100, 0],
        'Intensity D7': [100, 200, 300, 0, 200],
    })
    acc_map = {'P001': {'trend': 'Stable', 'description': 'ProtA'},
               'P002': {'trend': 'Degrading', 'description': 'ProtB'}}
    groups = {'D0': ['D0'], 'D7': ['D7']}
    result = coverage_kinetics(df, groups, acc_map)
    check("Cov: returns DF", isinstance(result, pd.DataFrame))
    check("Cov: has pep_change", 'pep_change' in result.columns)
    check("Cov: has trend", 'trend' in result.columns)
test_coverage_kinetics()

def test_sequence_composition():
    df = pd.DataFrame({
        'Proteins': ['P001']*5 + ['P002']*5,
        'Sequence': ['AVILMPPPK', 'LLLLLR', 'DDDDER', 'KKKKR', 'AAAAR',
                     'DDDDER', 'EEEER', 'KKKKNR', 'RRRRR', 'SSSSR'],
    })
    acc_map = {'P001': {'trend': 'Degrading', 'description': 'ProtA'},
               'P002': {'trend': 'Increasing', 'description': 'ProtB'}}
    result = sequence_composition(df, acc_map)
    check("Comp: returns DF", isinstance(result, pd.DataFrame))
    check("Comp: has GRAVY", 'GRAVY' in result.columns)
    check("Comp: has pct_Pro", 'pct_Pro' in result.columns)
    check("Comp: has trend", 'trend' in result.columns)
test_sequence_composition()

# ══ Summary ══
print(f"\n{'='*60}")
print(f"Test Results: {PASSED} passed, {FAILED} failed, {PASSED+FAILED} total")
print(f"{'='*60}")
if FAILED > 0:
    print("WARNING: Some tests FAILED!")
    sys.exit(1)
else:
    print("SUCCESS: All tests PASSED!")
    sys.exit(0)

