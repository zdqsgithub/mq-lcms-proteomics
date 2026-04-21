"""
MaxQuant LC-MS/MS Proteomics Core Module v2
=============================================
Data loading, filtering, transformation, QC.
Uses external allergen_db.json and taxonomy_db.json.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json, re
import warnings
warnings.filterwarnings('ignore')

_SKILL_DIR = Path(__file__).parent


def _load_json(name):
    path = _SKILL_DIR / name
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    return {}

_ALLERGEN_DB = None
_TAXONOMY_DB = None

def _get_allergen_db():
    global _ALLERGEN_DB
    if _ALLERGEN_DB is None:
        _ALLERGEN_DB = _load_json('allergen_db.json')
    return _ALLERGEN_DB

def _get_taxonomy_db():
    global _TAXONOMY_DB
    if _TAXONOMY_DB is None:
        _TAXONOMY_DB = _load_json('taxonomy_db.json')
    return _TAXONOMY_DB


# ── Data Loading ──────────────────────────────────────────────

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


# ── Filtering ─────────────────────────────────────────────────

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


# ── FASTA Parsing ─────────────────────────────────────────────

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


# ── Quantification ────────────────────────────────────────────

def get_quant_columns(df, groups, quant_type='iBAQ'):
    """Get quantification column names for each group."""
    prefix_map = {'iBAQ': 'iBAQ', 'lfq': 'LFQ intensity', 'intensity': 'Intensity'}
    prefix = prefix_map.get(quant_type, quant_type)
    cols = {}
    for g, samples in groups.items():
        cols[g] = [f'{prefix} {s}' for s in samples]
    return cols


def auto_detect_groups(pg, quant_type='iBAQ'):
    """Auto-detect sample groups from column names."""
    prefix_map = {'iBAQ': 'iBAQ ', 'lfq': 'LFQ intensity ', 'intensity': 'Intensity '}
    prefix = prefix_map.get(quant_type, 'iBAQ ')
    qcols = [c for c in pg.columns if c.startswith(prefix) and c != 'iBAQ peptides' and c != 'iBAQ']
    samples = [c[len(prefix):] for c in qcols]
    groups = {}
    for s in samples:
        parts = s.rsplit('-', 1)
        group = parts[0] if len(parts) > 1 and parts[1].isdigit() else s
        groups.setdefault(group, []).append(s)
    palette = ['#4E79A7','#E15759','#59A14F','#F28E2B','#B07AA1',
               '#76B7B2','#EDC948','#FF9DA7','#9C755F','#BAB0AC']
    colors = {g: palette[i % len(palette)] for i, g in enumerate(groups)}
    return groups, colors


# ── Transformation ────────────────────────────────────────────

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
        n_missing = result[col].isna().sum()
        if n_missing > 0:
            result.loc[result[col].isna(), col] = np.random.normal(
                col_mean - shift * col_std, col_std * scale, n_missing
            )
    return result


# ── QC Metrics ────────────────────────────────────────────────

def compute_qc_metrics(summary_df, groups=None):
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


# ── Allergen Annotation (v2: uses allergen_db.json) ──────────

def get_allergen_code(fasta_header, description):
    """Map protein to WHO/IUIS allergen nomenclature using external DB."""
    db = _get_allergen_db()
    fh = str(fasta_header) if pd.notna(fasta_header) else ''
    desc_lower = description.lower() if description else ''

    # 1. Try direct extraction from FASTA header (e.g. "Pen a 1.0102")
    match = re.search(r'\b([A-Z][a-z]{1,3}\s[a-z]\s\d+(?:\.\d+)*)\b', fh)
    if match:
        return re.sub(r'\.\d+$', '', match.group(1))

    # 2. Look up organism code from FASTA header
    org_map = db.get('organism_codes', {})
    org_match = re.search(r'_([A-Z]{5,6})\b', fh)
    org_code = org_match.group(1) if org_match else ''
    genus = org_map.get(org_code, '')

    # 3. Keyword -> allergen group from DB
    kw_groups = db.get('keyword_groups', {})
    for kw, info in kw_groups.items():
        if kw in desc_lower:
            grp = info if isinstance(info, str) else info.get('group', '')
            return f'{genus} {grp}' if genus else grp

    return ''


# ── Taxonomy Categorization (v2: uses taxonomy_db.json) ──────

def categorize_taxonomy(tax_name):
    """Categorize taxonomy name into biological group using external DB."""
    if not tax_name or tax_name == 'nan':
        return 'Unknown'
    db = _get_taxonomy_db()
    categories = db.get('categories', {})
    for category, genera in categories.items():
        if any(g in tax_name for g in genera):
            return category
    return 'Other'
