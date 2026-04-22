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


# ── Coverage Kinetics (v2.2) ─────────────────────────────

KD_HYDRO = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5,
            'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9,
            'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9,
            'Y': -1.3, 'V': 4.2}


def peptide_gravy(seq):
    """Compute GRAVY score (grand average of hydropathy)."""
    s = str(seq)
    return np.mean([KD_HYDRO.get(aa, 0) for aa in s]) if s else 0


def coverage_kinetics(pep_df, groups, acc_trend_map):
    """Track unique peptide count per protein per time point.

    Args:
        pep_df: peptides.txt DataFrame (filtered)
        groups: dict {group_name: [sample1, sample2, ...]}
        acc_trend_map: dict {accession: {'trend':..., 'description':...}}

    Returns:
        DataFrame with per-protein peptide count per TP, trend, and change.
    """
    def _map_trend(prots):
        for a in str(prots).split(';'):
            if a.strip() in acc_trend_map:
                return acc_trend_map[a.strip()].get('trend', 'Unknown')
        return 'Unknown'

    def _map_desc(prots):
        for a in str(prots).split(';'):
            if a.strip() in acc_trend_map:
                return acc_trend_map[a.strip()].get('description', str(prots)[:30])
        return str(prots)[:30]

    pep_df = pep_df.copy()
    pep_df['trend'] = pep_df['Proteins'].apply(_map_trend)
    pep_df['prot_desc'] = pep_df['Proteins'].apply(_map_desc)

    per_prot = {}
    group_names = list(groups.keys())
    for g, samps in groups.items():
        int_cols = [f'Intensity {s}' for s in samps if f'Intensity {s}' in pep_df.columns]
        if not int_cols:
            continue
        for desc, grp in pep_df.groupby('prot_desc'):
            detected = grp[(grp[int_cols].replace(0, np.nan) > 0).any(axis=1)]
            if desc not in per_prot:
                per_prot[desc] = {'trend': grp['trend'].iloc[0]}
            per_prot[desc][f'pep_{g}'] = len(detected)

    cov = pd.DataFrame(per_prot).T
    base, last = f'pep_{group_names[0]}', f'pep_{group_names[-1]}'
    cov['pep_change'] = cov.get(last, 0) - cov.get(base, 0)
    cov['pep_pct_change'] = np.where(
        cov.get(base, 0) > 0,
        cov['pep_change'] / cov[base].replace(0, np.nan) * 100, 0)

    return cov


def analyze_deamidation_sites(deam_path, groups, extract_desc_fn=None):
    """Analyze deamidation kinetics from Deamidation (NQ)Sites.txt.

    Same interface as analyze_oxidation_sites for consistency.
    """
    df = pd.read_csv(deam_path, sep='\t', low_memory=False)
    for col in ['Reverse', 'Potential contaminant']:
        if col in df.columns:
            df = df[df[col].fillna('').str.strip() != '+']

    if extract_desc_fn and 'Fasta headers' in df.columns:
        df['description'] = df['Fasta headers'].apply(extract_desc_fn)

    group_names = list(groups.keys())
    for g, samps in groups.items():
        ratio_cols = [f'Ratio mod/base {s}' for s in samps
                      if f'Ratio mod/base {s}' in df.columns]
        if ratio_cols:
            df[f'ratio_{g}'] = df[ratio_cols].replace(0, np.nan).mean(axis=1)

    base, last = group_names[0], group_names[-1]
    if f'ratio_{base}' in df.columns and f'ratio_{last}' in df.columns:
        df['ratio_change'] = df[f'ratio_{last}'] - df[f'ratio_{base}']

    return df


def sequence_composition(pep_df, acc_trend_map):
    """Compute per-protein sequence composition features.

    Returns DataFrame with GRAVY, aliphatic index, %Pro, %Met, %Cys, etc.
    """
    def _map_trend(prots):
        for a in str(prots).split(';'):
            if a.strip() in acc_trend_map:
                return acc_trend_map[a.strip()].get('trend', 'Unknown')
        return 'Unknown'

    pep_df = pep_df.copy()
    pep_df['trend'] = pep_df['Proteins'].apply(_map_trend)
    pep_df['GRAVY'] = pep_df['Sequence'].apply(peptide_gravy)

    feats = {}
    for desc, grp in pep_df.groupby(pep_df['Proteins'].apply(
        lambda x: next((acc_trend_map[a.strip()].get('description', str(x)[:30])
                       for a in str(x).split(';') if a.strip() in acc_trend_map), str(x)[:30]))):
        all_seq = ''.join(grp['Sequence'].dropna().values)
        if len(all_seq) < 20:
            continue
        n = len(all_seq)
        pct = {aa: all_seq.count(aa) / n for aa in 'AVILMFYWDENKRHSTCGPQ'}
        feats[desc] = {
            'trend': grp['trend'].iloc[0],
            'GRAVY': peptide_gravy(all_seq),
            'Aliphatic': (2.9*pct['A'] + 3.9*pct['V'] + 4.19*(pct['I']+pct['L'])) * 100,
            'pct_Pro': pct['P'] * 100,
            'pct_Met': pct['M'] * 100,
            'pct_Cys': pct['C'] * 100,
            'pct_charged': (pct['D']+pct['E']+pct['K']+pct['R']) * 100,
            'pct_hydrophobic': (pct['A']+pct['V']+pct['I']+pct['L']+pct['F']+pct['W']+pct['M']) * 100,
        }

    return pd.DataFrame(feats).T


# ── Fragment Profiling (v2.2.3) ───────────────────────────

CALPAIN_AA = {'L', 'V', 'I', 'F', 'A', 'M', 'Y', 'W'}
CASPASE_AA = {'D', 'E'}

def fragment_profiling(pep_df, groups, acc_trend_map):
    """Full protease fragment profiling from semi-specific search data.

    Returns dict with:
      - summary: {total, fully_tryptic, semi_tryptic, non_tryptic, protease_fragments}
      - kinetics: per-TP counts {group: {n_full, n_semi, n_protease, pct_protease_int}}
      - p1_specificity: {calpain_n, caspase_n, other_n, aa_counts: Series}
      - new_fragments_df: DataFrame of fragments appearing only at last TP
      - lost_count: int
      - per_protein_df: DataFrame of per-protein cleavage counts
      - true_protease_df: DataFrame of non-K/R/M semi-tryptic peptides
    """
    pep_ann = detect_semi_tryptic(pep_df)

    # Map trends via accession
    def get_trend(prots):
        for a in str(prots).split(';'):
            a = a.strip()
            if a in acc_trend_map:
                return acc_trend_map[a].get('trend', 'Unknown')
        return 'Unknown'

    def get_desc(prots):
        for a in str(prots).split(';'):
            a = a.strip()
            if a in acc_trend_map:
                return acc_trend_map[a].get('description', str(prots)[:30])
        return str(prots)[:30]

    pep_ann['protein_trend'] = pep_ann['Proteins'].apply(get_trend)
    pep_ann['prot_desc'] = pep_ann['Proteins'].apply(get_desc)

    semi = pep_ann[pep_ann['cleavage_type'] == 'Semi-tryptic'].copy()
    true_protease = semi[~semi['Amino acid before'].isin(['K', 'R', 'M', '-', '', 'nan', np.nan])].copy()

    n_full = (pep_ann['cleavage_type'] == 'Fully tryptic').sum()
    n_semi = len(semi)
    n_nontryp = (pep_ann['cleavage_type'] == 'Non-tryptic').sum()

    summary = {'total': len(pep_ann), 'fully_tryptic': n_full,
               'semi_tryptic': n_semi, 'non_tryptic': n_nontryp,
               'protease_fragments': len(true_protease)}

    # Per-TP kinetics
    group_names = list(groups.keys())
    kinetics = {}
    for g, samps in groups.items():
        int_cols = [f'Intensity {s}' for s in samps if f'Intensity {s}' in pep_ann.columns]
        if not int_cols:
            continue
        detected = pep_ann[(pep_ann[int_cols].replace(0, np.nan) > 0).any(axis=1)]
        semi_det = detected[detected['cleavage_type'] == 'Semi-tryptic']
        full_det = detected[detected['cleavage_type'] == 'Fully tryptic']
        prot_det = semi_det[~semi_det['Amino acid before'].isin(['K', 'R', 'M', '-', '', 'nan'])]
        semi_int = semi_det[int_cols].replace(0, np.nan).sum().sum()
        full_int = full_det[int_cols].replace(0, np.nan).sum().sum()
        prot_int = prot_det[int_cols].replace(0, np.nan).sum().sum()
        total = semi_int + full_int
        kinetics[g] = {
            'n_full': len(full_det), 'n_semi': len(semi_det),
            'n_protease': len(prot_det),
            'pct_protease_int': prot_int / total * 100 if total > 0 else 0,
        }

    # P1 cleavage specificity
    if len(true_protease) > 0:
        aa_counts = true_protease['Amino acid before'].value_counts()
        calpain_n = sum(aa_counts.get(aa, 0) for aa in CALPAIN_AA)
        caspase_n = sum(aa_counts.get(aa, 0) for aa in CASPASE_AA)
        other_n = len(true_protease) - calpain_n - caspase_n
    else:
        aa_counts = pd.Series(dtype=int)
        calpain_n = caspase_n = other_n = 0

    p1 = {'calpain_n': calpain_n, 'caspase_n': caspase_n,
           'other_n': other_n, 'aa_counts': aa_counts}

    # New / lost fragments
    d0_cols = [f'Intensity {s}' for s in groups[group_names[0]]
               if f'Intensity {s}' in pep_ann.columns]
    d_last_cols = [f'Intensity {s}' for s in groups[group_names[-1]]
                   if f'Intensity {s}' in pep_ann.columns]

    if len(true_protease) > 0 and d0_cols and d_last_cols:
        at_d0 = set(true_protease[(true_protease[d0_cols].replace(0, np.nan) > 0).any(axis=1)].index)
        at_last = set(true_protease[(true_protease[d_last_cols].replace(0, np.nan) > 0).any(axis=1)].index)
        new_idx = at_last - at_d0
        lost_count = len(at_d0 - at_last)
        new_fragments_df = true_protease.loc[list(new_idx)] if len(new_idx) > 0 else pd.DataFrame()
    else:
        new_fragments_df = pd.DataFrame()
        lost_count = 0

    # Per-protein cleavage profiling
    if len(true_protease) > 0:
        per_prot = true_protease.groupby('prot_desc').agg(
            n_fragments=('Sequence', 'count'),
            protein_trend=('protein_trend', 'first'),
            calpain_like=('Amino acid before', lambda x: sum(str(a) in CALPAIN_AA for a in x))
        ).sort_values('n_fragments', ascending=False)
        total_per = pep_ann.groupby('prot_desc').size()
        per_prot['n_total'] = total_per.reindex(per_prot.index).fillna(0)
        per_prot['pct_clipped'] = per_prot['n_fragments'] / per_prot['n_total'] * 100
    else:
        per_prot = pd.DataFrame()

    return {
        'summary': summary,
        'kinetics': kinetics,
        'p1_specificity': p1,
        'new_fragments_df': new_fragments_df,
        'lost_count': lost_count,
        'per_protein_df': per_prot,
        'true_protease_df': true_protease,
    }


# ── Biophysical Property Analysis ─────────────────────────

# Kyte-Doolittle hydropathy scale
_KD = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,
       'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,
       'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}

# Amino acid molecular weights (Da)
_MW_AA = {'A':89.09,'R':174.20,'N':132.12,'D':133.10,'C':121.16,'Q':146.15,
          'E':147.13,'G':75.03,'H':155.16,'I':131.17,'L':131.17,'K':146.19,
          'M':149.21,'F':165.19,'P':115.13,'S':105.09,'T':119.12,'W':204.23,
          'Y':181.19,'V':117.15}


def _compute_pI(seq, aa_count):
    """Compute isoelectric point by binary search on Henderson-Hasselbalch."""
    pos_pKa = {'K': 10.5, 'R': 12.4, 'H': 6.0}
    neg_pKa = {'D': 3.9, 'E': 4.1, 'C': 8.3, 'Y': 10.1}
    def charge_at_pH(pH):
        charge = 1.0 / (1 + 10**(pH - 9.7))   # N-term
        charge -= 1.0 / (1 + 10**(2.3 - pH))   # C-term
        for aa, pKa in pos_pKa.items():
            charge += aa_count.get(aa, 0) / (1 + 10**(pH - pKa))
        for aa, pKa in neg_pKa.items():
            charge -= aa_count.get(aa, 0) / (1 + 10**(pKa - pH))
        return charge
    lo, hi = 0, 14
    for _ in range(100):
        mid = (lo + hi) / 2
        if charge_at_pH(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def compute_protein_properties(seq):
    """Compute biophysical properties from an amino acid sequence string.
    
    Returns dict with: length, MW_kDa, GRAVY, Aliphatic_Index, pI,
    pct_hydrophobic, pct_charged, pct_aromatic, pct_Cys, pct_Pro, pct_Met,
    net_charge_pH7, aggregation_score.
    """
    n = len(seq)
    if n == 0:
        return {}
    aa_count = {aa: seq.count(aa) for aa in 'ACDEFGHIKLMNPQRSTVWY'}
    aa_pct = {aa: count / n * 100 for aa, count in aa_count.items()}
    gravy = sum(_KD.get(aa, 0) for aa in seq) / n
    aliphatic = aa_pct.get('A', 0) + 2.9*aa_pct.get('V', 0) + 3.9*(aa_pct.get('I', 0) + aa_pct.get('L', 0))
    mw = sum(_MW_AA.get(aa, 110) for aa in seq) - (n-1)*18.015
    pI = _compute_pI(seq, aa_count)
    hydrophobic = sum(aa_count.get(aa, 0) for aa in 'AVILMFW') / n * 100
    charged = sum(aa_count.get(aa, 0) for aa in 'DEKR') / n * 100
    aromatic = sum(aa_count.get(aa, 0) for aa in 'FWY') / n * 100
    # Net charge at pH 7
    pos_pKa = {'K': 10.5, 'R': 12.4, 'H': 6.0}
    neg_pKa = {'D': 3.9, 'E': 4.1, 'C': 8.3, 'Y': 10.1}
    net_charge = 1.0 / (1 + 10**(7 - 9.7)) - 1.0 / (1 + 10**(2.3 - 7))
    for aa, pKa in pos_pKa.items():
        net_charge += aa_count.get(aa, 0) / (1 + 10**(7 - pKa))
    for aa, pKa in neg_pKa.items():
        net_charge -= aa_count.get(aa, 0) / (1 + 10**(pKa - 7))
    agg_score = gravy * 10 - abs(net_charge) * 0.5 + hydrophobic * 0.1
    return {
        'length': n, 'MW_kDa': round(mw / 1000, 1), 'GRAVY': round(gravy, 3),
        'Aliphatic_Index': round(aliphatic, 1), 'pI': round(pI, 2),
        'pct_hydrophobic': round(hydrophobic, 1), 'pct_charged': round(charged, 1),
        'pct_aromatic': round(aromatic, 1), 'pct_Cys': round(aa_pct.get('C', 0), 2),
        'pct_Pro': round(aa_pct.get('P', 0), 2), 'pct_Met': round(aa_pct.get('M', 0), 2),
        'net_charge_pH7': round(net_charge, 1), 'aggregation_score': round(agg_score, 2),
    }


def biophysical_analysis(stab_df, output_dir):
    """Fetch UniProt sequences and compute biophysical properties for all proteins.
    
    Compares Degrading vs Stable vs Increasing groups with Mann-Whitney U tests.
    Returns (results_df, comparison_df, fasta_path_or_None).
    """
    import urllib.request

    # Extract primary accession per protein
    def primary_acc(acc_str):
        return str(acc_str).split(';')[0].strip() if pd.notna(acc_str) else ''

    stab_df = stab_df.copy()
    stab_df['primary_acc'] = stab_df['Majority protein IDs'].apply(primary_acc)
    all_accs = [a for a in stab_df['primary_acc'].dropna().unique() if len(a) > 3]

    # Fetch from UniProt
    sequences = {}
    for acc in all_accs:
        try:
            url = f"https://rest.uniprot.org/uniprotkb/{acc}.fasta"
            req = urllib.request.Request(url, headers={'User-Agent': 'Python/MaxQuantSkill'})
            with urllib.request.urlopen(req, timeout=10) as resp:
                fasta = resp.read().decode('utf-8')
                lines = fasta.strip().split('\n')
                sequences[acc] = {'header': lines[0], 'sequence': ''.join(lines[1:])}
        except Exception:
            pass

    # Save FASTA
    biophys_dir = Path(output_dir) / 'biophysical'
    biophys_dir.mkdir(exist_ok=True)
    fasta_path = biophys_dir / 'proteins.fasta'
    with open(fasta_path, 'w') as f:
        for acc, data in sequences.items():
            f.write(f"{data['header']}\n")
            seq = data['sequence']
            for i in range(0, len(seq), 70):
                f.write(seq[i:i+70] + '\n')

    # Compute properties
    results = []
    for _, row in stab_df.iterrows():
        acc = row['primary_acc']
        entry = {'accession': acc,
                 'description': str(row.get('description', ''))[:50],
                 'trend': row.get('trend', 'Unknown')}
        if acc in sequences:
            entry.update(compute_protein_properties(sequences[acc]['sequence']))
        results.append(entry)
    results_df = pd.DataFrame(results)
    results_df.to_csv(biophys_dir / 'protein_properties.csv', index=False)

    # Statistical comparison
    features = ['MW_kDa', 'GRAVY', 'Aliphatic_Index', 'pI', 'pct_hydrophobic',
                'pct_charged', 'pct_aromatic', 'pct_Cys', 'pct_Pro', 'pct_Met',
                'net_charge_pH7', 'length', 'aggregation_score']
    comparison_rows = []
    for feat in features:
        row_data = {'Feature': feat}
        for trend in ['Degrading', 'Stable', 'Increasing']:
            vals = results_df[results_df['trend'] == trend][feat].dropna()
            if len(vals) > 0:
                row_data[f'{trend}_mean'] = round(vals.mean(), 3)
                row_data[f'{trend}_std'] = round(vals.std(), 3)
                row_data[f'{trend}_n'] = len(vals)
        d_vals = results_df[results_df['trend'] == 'Degrading'][feat].dropna()
        s_vals = results_df[results_df['trend'] == 'Stable'][feat].dropna()
        if len(d_vals) >= 3 and len(s_vals) >= 3:
            try:
                _, p = sp_stats.mannwhitneyu(d_vals, s_vals)
                row_data['p_DvS'] = round(p, 4)
            except Exception:
                row_data['p_DvS'] = np.nan
        comparison_rows.append(row_data)
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(biophys_dir / 'biophysical_comparison.csv', index=False)

    return results_df, comparison_df, fasta_path

