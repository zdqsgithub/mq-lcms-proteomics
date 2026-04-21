"""
MaxQuant LC-MS/MS Proteomics Core Module
=========================================
Data loading, filtering, transformation, and QC.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import re
import warnings
warnings.filterwarnings('ignore')


def load_maxquant(data_dir, files=None):
    """Load MaxQuant output files from a directory."""
    data_dir = Path(data_dir)
    result = {}
    default_files = {
        'proteinGroups': 'proteinGroups.txt',
        'peptides': 'peptides.txt',
        'evidence': 'evidence.txt',
        'summary': 'summary.txt',
        'parameters': 'parameters.txt',
    }
    targets = files or default_files
    for key, fname in targets.items():
        fpath = data_dir / fname
        if fpath.exists():
            result[key] = pd.read_csv(fpath, sep='\t', low_memory=False)
            print(f"  Loaded {fname}: {len(result[key])} rows")
        else:
            print(f"  {fname} not found, skipping")
    return result


def load_metadata(meta_path):
    """Load experimental metadata (SDRF or CSV)."""
    meta_path = Path(meta_path)
    sep = '\t' if meta_path.suffix in ['.tsv', '.sdrf'] else ','
    return pd.read_csv(meta_path, sep=sep)


def filter_protein_groups(df):
    """Standard MaxQuant filtering: remove reverse, contaminant, site-only."""
    mask = pd.Series(True, index=df.index)
    for col in ['Reverse', 'Potential contaminant', 'Only identified by site']:
        if col in df.columns:
            mask &= df[col].fillna('').str.strip() != '+'
    filtered = df[mask].copy()
    print(f"  Filtered: {len(df)} -> {len(filtered)} protein groups")
    return filtered


def filter_peptides(df):
    """Filter peptides: remove reverse and contaminants."""
    mask = pd.Series(True, index=df.index)
    for col in ['Reverse', 'Potential contaminant']:
        if col in df.columns:
            mask &= df[col].fillna('').str.strip() != '+'
    return df[mask].copy()


def extract_description(fasta_header):
    """Extract readable protein description from UniProt FASTA headers."""
    if pd.isna(fasta_header) or str(fasta_header).strip() == '':
        return ''
    first = str(fasta_header).split(';')[0].strip()
    parts = first.split(' ', 1)
    if len(parts) < 2:
        return first
    desc = parts[1]
    if ' OS=' in desc:
        desc = desc.split(' OS=')[0]
    return desc.strip()


def get_quant_columns(df, groups, quant_type='iBAQ'):
    """Get quantification column names for each group."""
    prefix_map = {'iBAQ': 'iBAQ', 'lfq': 'LFQ intensity', 'intensity': 'Intensity'}
    prefix = prefix_map.get(quant_type, quant_type)
    cols = {}
    for g, samples in groups.items():
        cols[g] = [f'{prefix} {s}' for s in samples]
    return cols


def log2_transform(df, columns):
    """Log2 transform, replacing 0 with NaN."""
    result = df[columns].copy()
    result = result.replace(0, np.nan)
    return np.log2(result)


def impute_missing(df, shift=1.8, scale=0.3):
    """Down-shifted Gaussian imputation for MNAR data."""
    result = df.copy()
    for col in result.columns:
        valid = result[col].dropna()
        if len(valid) == 0:
            continue
        col_mean = valid.mean()
        col_std = valid.std()
        if col_std == 0 or np.isnan(col_std):
            col_std = 1.0
        imp_mean = col_mean - shift * col_std
        imp_std = col_std * scale
        n_missing = result[col].isna().sum()
        if n_missing > 0:
            result.loc[result[col].isna(), col] = np.random.normal(
                imp_mean, imp_std, n_missing
            )
    return result


def compute_qc_metrics(summary_df, groups):
    """Extract per-sample QC metrics from summary.txt."""
    if summary_df is None:
        return None
    data = summary_df[summary_df['Raw file'] != 'Total'].copy()
    data = data[data['Raw file'].notna()]
    metrics = []
    for _, row in data.iterrows():
        metrics.append({
            'sample': row.get('Experiment', row.get('Raw file', '')),
            'msms_submitted': row.get('MS/MS submitted', 0),
            'msms_identified': row.get('MS/MS identified', 0),
            'id_rate': row.get('MS/MS identified [%]', 0),
            'peptides': row.get('Peptide sequences identified', 0),
        })
    return pd.DataFrame(metrics)


def get_allergen_code(fasta_header, description):
    """Map protein to WHO/IUIS allergen nomenclature."""
    fh = str(fasta_header) if pd.notna(fasta_header) else ''
    desc_lower = description.lower() if description else ''
    
    # Try direct extraction from FASTA header
    match = re.search(r'\b([A-Z][a-z]{1,3}\s[a-z]\s\d+(?:\.\d+)*)\b', fh)
    if match:
        code = match.group(1)
        code = re.sub(r'\.\d+$', '', code)
        return code
    
    # Organism code mapping
    org_map = {
        'PENVA': 'Pen v', 'PENMO': 'Pen m', 'PENAT': 'Pen a',
        'DERPT': 'Der p', 'PANBO': 'Pan b', 'MACRS': 'Mac r',
        'CRACN': 'Cra c', 'METEN': 'Met e',
    }
    org_match = re.search(r'_([A-Z]{5,6})\b', fh)
    org_code = org_match.group(1) if org_match else ''
    genus = org_map.get(org_code, '')
    
    # Keyword -> allergen group
    group_map = {
        'tropomyosin': '1', 'arginine kinase': '2',
        'myosin light chain': '3', 'sarcoplasmic calcium': '4',
        'paramyosin': '11', 'hemocyanin': '3 (Hemo)',
        'triosephosphate isomerase': '8', 'enolase': '10',
    }
    for kw, grp in group_map.items():
        if kw in desc_lower:
            return f'{genus} {grp}' if genus else grp
    return ''


def categorize_taxonomy(tax_name):
    """Categorize taxonomy name into biological group."""
    if not tax_name or tax_name == 'nan':
        return 'Unknown'
    shrimp_genera = ['Penaeus', 'Macrobrachium', 'Artemia', 'Halocaridina', 'Thor']
    bacteria_genera = ['Vibrio', 'Planococcus', 'Bacillus', 'Photobacterium']
    if any(g in tax_name for g in shrimp_genera):
        return 'Shrimp/Crustacean'
    if 'Dermatophagoides' in tax_name:
        return 'Dust Mite'
    if any(g in tax_name for g in bacteria_genera):
        return 'Bacteria'
    return 'Other'
