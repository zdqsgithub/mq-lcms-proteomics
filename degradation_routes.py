"""
MaxQuant LC-MS/MS Proteomics - Degradation Route Analysis Module v2.1
=====================================================================
Functional categorization, oxidation kinetics, protease/phosphatase
inventory, semi-tryptic peptide detection, deamidation assessment.
"""
import pandas as pd
import numpy as np
from scipy import stats as sp_stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# ── Functional Category Assignment ────────────────────────

FUNCTIONAL_CATEGORIES = {
    'Chaperone/HSP': ['heat shock', 'hsp', 'chaperone', 'chaperonin', 'glucose-regulated protein'],
    'Redox/Antioxidant': ['glutathione', 'superoxide dismutase', 'peroxiredoxin', 'thioredoxin',
                         'catalase', 'glutathione peroxidase', 'peroxidase'],
    'Protein Folding/Processing': ['peptidyl-prolyl', 'protein disulfide-isomerase', 'calreticulin',
                                   'prefoldin', 'nascent polypeptide'],
    'Ubiquitin/Proteasome': ['ubiquitin', 'proteasome', 'sumo'],
    'Translation/Ribosome': ['elongation factor', 'translation initiation', 'ribosomal',
                             'translationally-controlled'],
    'Signaling/Kinase': ['kinase', 'phosphatase', 'calmodulin', '14-3-3', 'rab', 'gtp-binding',
                        'signal transducer'],
    'Structural/Muscle': ['tropomyosin', 'myosin', 'actin', 'troponin', 'connectin', 'paramyosin'],
    'Immune/Defense': ['hemocyanin', 'prophenoloxidase', 'crustin', 'antimicrobial', 'clottable',
                      'hemolymph', 'l-dopachrome', 'masquerade', 'clip domain', 'pacifastin',
                      'penlectin', 'lectin'],
    'Metabolic Enzyme': ['synthase', 'dehydrogenase', 'isomerase', 'transferase', 'lyase',
                        'phosphorylase', 'methyltransferase', 'nuclease', 'carbonic anhydrase',
                        'enolase', 'atp synthase'],
    'Lipid Binding/Transport': ['fatty acid binding', 'acyl-coa', 'crustacyanin', 'lipid transfer'],
    'Calcium Binding': ['sarcoplasmic calcium', 'calcium-binding'],
    'Vesicle/Transport': ['vesicle-fusing', 'ap complex', 'gdp dissociation',
                         'sodium/potassium-transporting'],
    'Viral/WSSV': ['wsv', 'vp466', 'non-structural protein'],
}

PROTEASE_KEYWORDS = ['protease', 'proteinase', 'peptidase', 'calpain', 'cathepsin',
                     'caspase', 'metalloproteinase', 'carboxypeptidase', 'aminopeptidase',
                     'trypsin', 'chymotrypsin', 'dipeptidyl', 'prophenoloxidase',
                     'clip domain', 'masquerade']
PHOSPHATASE_KEYWORDS = ['phosphatase', 'phosphoprotein']


def assign_functional_category(description, categories=None):
    """Assign functional category based on protein description keywords."""
    if categories is None:
        categories = FUNCTIONAL_CATEGORIES
    d = str(description).lower()
    for cat, keywords in categories.items():
        if any(kw in d for kw in keywords):
            return cat
    return 'Other'


def functional_enrichment(tc_df, description_col='description'):
    """Compute functional category enrichment across trend groups."""
    tc_df = tc_df.copy()
    tc_df['functional_category'] = tc_df[description_col].apply(assign_functional_category)

    trend_groups = {}
    for trend in ['Degrading', 'Stable', 'Increasing']:
        trend_groups[trend] = tc_df[tc_df['trend'] == trend]

    cat_counts = pd.DataFrame()
    for trend, df in trend_groups.items():
        cat_counts[trend] = df['functional_category'].value_counts()
    cat_counts = cat_counts.fillna(0).astype(int)
    cat_counts['Total'] = cat_counts.sum(axis=1)
    cat_counts = cat_counts.sort_values('Total', ascending=False)

    cat_pcts = pd.DataFrame()
    for trend in ['Degrading', 'Stable', 'Increasing']:
        n = len(trend_groups[trend])
        cat_pcts[trend] = (cat_counts[trend] / n * 100).round(1) if n > 0 else 0

    return cat_counts, cat_pcts, trend_groups


# ── Oxidation Site Analysis ───────────────────────────────

def analyze_oxidation_sites(ox_path, groups, extract_desc_fn=None):
    """Analyze methionine oxidation kinetics from Oxidation (M)Sites.txt.

    Args:
        ox_path: Path to Oxidation (M)Sites.txt
        groups: dict {group_name: [sample1, sample2, ...]}
        extract_desc_fn: function to parse FASTA headers

    Returns:
        DataFrame with per-site oxidation ratios and kinetics
    """
    ox = pd.read_csv(ox_path, sep='\t', low_memory=False)
    for col in ['Reverse', 'Potential contaminant']:
        if col in ox.columns:
            ox = ox[ox[col].fillna('').str.strip() != '+']

    if extract_desc_fn and 'Fasta headers' in ox.columns:
        ox['description'] = ox['Fasta headers'].apply(extract_desc_fn)

    group_names = list(groups.keys())
    # Compute mean ratio and intensity per group
    for g, samps in groups.items():
        ratio_cols = [f'Ratio mod/base {s}' for s in samps
                      if f'Ratio mod/base {s}' in ox.columns]
        int_cols = [f'Intensity {s}' for s in samps
                    if f'Intensity {s}' in ox.columns]
        if ratio_cols:
            ox[f'ratio_{g}'] = ox[ratio_cols].replace(0, np.nan).mean(axis=1)
        if int_cols:
            ox[f'int_{g}'] = ox[int_cols].replace(0, np.nan).mean(axis=1)

    # Compute change vs baseline
    base = group_names[0]
    last = group_names[-1]
    if f'ratio_{base}' in ox.columns and f'ratio_{last}' in ox.columns:
        ox['ratio_change'] = ox[f'ratio_{last}'] - ox[f'ratio_{base}']
        ox['ratio_fc'] = np.where(
            (ox[f'ratio_{base}'] > 0) & (ox[f'ratio_{last}'] > 0),
            ox[f'ratio_{last}'] / ox[f'ratio_{base}'], np.nan)

    return ox


def correlate_oxidation_degradation(ox_df, stab_df, group_names):
    """Correlate per-protein oxidation with degradation."""
    last = group_names[-1]
    # Aggregate oxidation per protein
    if 'description' not in ox_df.columns:
        return None, None, None

    prot_ox = ox_df.groupby('description').agg({
        f'ratio_{group_names[0]}': 'mean',
        f'ratio_{last}': 'mean',
    }).dropna()

    # Find matching FC column in stability data
    fc_col = None
    for candidate in [f'log2FC_{last}', f'log2FC_{group_names[-1]}']:
        if candidate in stab_df.columns:
            fc_col = candidate
            break
    if fc_col is None:
        return None, None, None

    prot_ox['desc_key'] = prot_ox.index.str[:30]
    stab_df = stab_df.copy()
    stab_df['desc_key'] = stab_df['description'].astype(str).str[:30]

    merged = prot_ox.merge(stab_df[['desc_key', fc_col]], on='desc_key', how='inner')
    if len(merged) < 4:
        return merged, None, None

    r_val, p_val = sp_stats.pearsonr(
        merged[f'ratio_{last}'].values,
        merged[fc_col].values
    )
    return merged, r_val, p_val


# ── Protease / Semi-Tryptic Analysis ─────────────────────

def detect_semi_tryptic(peptides_df):
    """Classify peptides as fully tryptic, semi-tryptic, or non-tryptic."""
    df = peptides_df.copy()

    def _classify(row):
        aa_b = str(row.get('Amino acid before', '')).strip()
        seq = str(row.get('Sequence', ''))
        aa_a = str(row.get('Amino acid after', '')).strip()
        n_ok = aa_b in ['K', 'R', '-', '', 'nan']
        c_ok = (seq[-1] in ['K', 'R'] if seq else False) or aa_a in ['-', '', 'nan']
        if n_ok and c_ok:
            return 'Fully tryptic'
        elif n_ok or c_ok:
            return 'Semi-tryptic'
        else:
            return 'Non-tryptic'

    df['cleavage_type'] = df.apply(_classify, axis=1)
    return df


def semi_tryptic_kinetics(peptides_df, groups):
    """Compute semi-tryptic peptide fraction per time point."""
    df = detect_semi_tryptic(peptides_df)
    semi = df[df['cleavage_type'] == 'Semi-tryptic']
    full = df[df['cleavage_type'] == 'Fully tryptic']

    ratios = {}
    for g, samps in groups.items():
        int_cols = [f'Intensity {s}' for s in samps if f'Intensity {s}' in df.columns]
        if not int_cols:
            continue
        semi_int = semi[int_cols].replace(0, np.nan).sum().sum()
        full_int = full[int_cols].replace(0, np.nan).sum().sum()
        total = semi_int + full_int
        ratios[g] = semi_int / total * 100 if total > 0 else 0

    return ratios, df


def inventory_proteases_phosphatases(stab_df, protease_kw=None, phosphatase_kw=None):
    """Identify proteases and phosphatases in the stability summary."""
    if protease_kw is None:
        protease_kw = PROTEASE_KEYWORDS
    if phosphatase_kw is None:
        phosphatase_kw = PHOSPHATASE_KEYWORDS

    proteases = []
    phosphatases = []

    for _, r in stab_df.iterrows():
        desc = str(r.get('description', '')).lower()
        trend = r.get('trend', 'Unknown')
        if any(kw in desc for kw in protease_kw):
            risk = ('HIGH' if trend == 'Increasing' else
                    'MODERATE' if trend == 'Stable' else 'LOW')
            proteases.append({**r.to_dict(), 'risk': risk})
        if any(kw in desc for kw in phosphatase_kw):
            phosphatases.append(r.to_dict())

    return pd.DataFrame(proteases), pd.DataFrame(phosphatases)


def peptide_appearance(peptides_df, groups):
    """Track peptide appearance/disappearance across time points."""
    present = {}
    for g, samps in groups.items():
        int_cols = [f'Intensity {s}' for s in samps if f'Intensity {s}' in peptides_df.columns]
        if int_cols:
            present[g] = set(peptides_df[(peptides_df[int_cols].replace(0, np.nan) > 0).any(axis=1)].index)
        else:
            present[g] = set()

    group_names = list(groups.keys())
    baseline = group_names[0]
    last = group_names[-1]

    lost = present.get(baseline, set()) - present.get(last, set())
    gained = present.get(last, set()) - present.get(baseline, set())

    return {
        'lost_from_baseline': len(lost),
        'gained_at_last': len(gained),
        'lost_indices': lost,
        'gained_indices': gained,
        'present_per_group': {g: len(s) for g, s in present.items()},
    }


def count_deamidation_motifs(peptides_df):
    """Count deamidation-prone motifs (NG, NS, NT) in peptide sequences."""
    count = 0
    for seq in peptides_df['Sequence'].dropna():
        s = str(seq)
        count += s.count('NG') + s.count('NS') + s.count('NT')
    return count
