"""
MaxQuant LC-MS/MS Proteomics Bioinformatics & Modeling Skill
=============================================================
Main entry point. Orchestrates the full pipeline:
  1. Load & filter data
  2. QC metrics
  3. Differential abundance
  4. Taxonomy enrichment
  5. Allergen annotation
  6. Visualization
  7. Predictive modeling (optional)
  8. Report generation + reproducibility bundle

Usage:
  python maxquant_lcms_skill.py --demo --output demo_report
  python maxquant_lcms_skill.py --input proteinGroups.txt --metadata sdrf.tsv --output report
"""
import sys, os, argparse, hashlib, json
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd

# Add skill directory to path
sys.path.insert(0, str(Path(__file__).parent))
from core import (load_maxquant, load_metadata, filter_protein_groups,
                  extract_description, get_quant_columns, log2_transform,
                  impute_missing, compute_qc_metrics, get_allergen_code,
                  categorize_taxonomy)
from stats_engine import (differential_abundance, classify_significance,
                        run_pca, train_classifier, compute_replicate_correlation,
                        benjamini_hochberg)
from visualization import (plot_msms_summary, plot_protein_counts,
                           plot_missing_values, plot_intensity_distribution,
                           plot_replicate_correlation, plot_venn, plot_volcano,
                           plot_allergen_heatmap, plot_pca, plot_top_proteins)


ALLERGEN_KEYWORDS = [
    'tropomyosin', 'arginine kinase', 'hemocyanin', 'myosin',
    'actin', 'sarcoplasmic calcium', 'aldehyde dehydrogenase',
    'triosephosphate isomerase', 'glyceraldehyde', 'enolase',
    'paramyosin', 'glutathione', 'peroxiredoxin', 'heat shock',
    'superoxide dismutase', 'crustacyanin',
]


def generate_demo_data():
    """Generate minimal demo proteinGroups for testing."""
    np.random.seed(42)
    n = 100
    demo = pd.DataFrame({
        'Protein IDs': [f'P{i:05d}' for i in range(n)],
        'Majority protein IDs': [f'P{i:05d}' for i in range(n)],
        'Protein names': [f'Protein_{i}' for i in range(n)],
        'Gene names': [f'GENE{i}' for i in range(n)],
        'Fasta headers': [f'sp|P{i:05d}|PROT{i}_HUMAN Protein {i} OS=Homo sapiens' for i in range(n)],
        'Reverse': [''] * n,
        'Potential contaminant': [''] * (n-5) + ['+'] * 5,
        'Only identified by site': [''] * n,
        'Peptides': np.random.randint(1, 20, n),
        'Unique peptides': np.random.randint(1, 15, n),
        'Sequence coverage [%]': np.random.uniform(5, 80, n),
        'Mol. weight [kDa]': np.random.uniform(10, 200, n),
        'iBAQ A-1': np.random.lognormal(20, 3, n).astype(int),
        'iBAQ A-2': np.random.lognormal(20, 3, n).astype(int),
        'iBAQ B-1': np.random.lognormal(19, 3, n).astype(int),
        'iBAQ B-2': np.random.lognormal(19, 3, n).astype(int),
    })
    # Inject some real differences
    demo.loc[:9, 'iBAQ A-1'] *= 10
    demo.loc[:9, 'iBAQ A-2'] *= 10
    demo.loc[:4, 'Fasta headers'] = [
        'sp|P00001|TROP_PENVA Tropomyosin Pen a 1 OS=Penaeus vannamei',
        'sp|P00002|ARGK_PENVA Arginine kinase OS=Penaeus vannamei',
        'sp|P00003|MYOL_DERPT Myosin light chain Der p 2 OS=Dermatophagoides pteronyssinus',
        'sp|P00004|HEMO_PENVA Hemocyanin subunit OS=Penaeus vannamei',
        'sp|P00005|ENOL_PENVA Enolase OS=Penaeus vannamei',
    ]
    demo.loc[:4, 'Taxonomy names'] = [
        'Penaeus vannamei', 'Penaeus vannamei',
        'Dermatophagoides pteronyssinus', 'Penaeus vannamei', 'Penaeus vannamei'
    ]
    groups = {'GroupA': ['A-1', 'A-2'], 'GroupB': ['B-1', 'B-2']}
    colors = {'GroupA': '#4E79A7', 'GroupB': '#E15759'}
    return demo, groups, colors


def auto_detect_groups(pg):
    """Auto-detect sample groups from iBAQ/LFQ column names."""
    import re
    ibaq_cols = [c for c in pg.columns if c.startswith('iBAQ ')]
    if not ibaq_cols:
        ibaq_cols = [c for c in pg.columns if c.startswith('LFQ intensity ')]
    samples = [c.split(' ', 1)[-1] if ' ' in c else c for c in ibaq_cols]
    # Group by common prefix (before last dash-number)
    groups = {}
    for s in samples:
        parts = s.rsplit('-', 1)
        group = parts[0] if len(parts) > 1 and parts[1].isdigit() else s
        groups.setdefault(group, []).append(s)
    # Generate colors
    palette = ['#4E79A7','#E15759','#59A14F','#F28E2B','#B07AA1',
               '#76B7B2','#EDC948','#FF9DA7','#9C755F','#BAB0AC']
    colors = {g: palette[i % len(palette)] for i, g in enumerate(groups)}
    return groups, colors


def write_checksums(output_dir):
    """SHA-256 checksums for all output files."""
    output_dir = Path(output_dir)
    lines = []
    for f in sorted(output_dir.rglob('*')):
        if f.is_file() and f.name != 'checksums.sha256':
            h = hashlib.sha256(f.read_bytes()).hexdigest()
            lines.append(f'{h}  {f.relative_to(output_dir)}')
    (output_dir / 'checksums.sha256').write_text('\n'.join(lines), encoding='utf-8')


def write_commands(output_dir, args):
    """Write commands.sh for reproducibility."""
    cmd = f"python {Path(__file__).name}"
    for k, v in vars(args).items():
        if v is not None and v is not False and k != 'func':
            cmd += f" --{k.replace('_','-')} {v}"
    (Path(output_dir) / 'commands.sh').write_text(
        f"#!/bin/bash\n# Generated {datetime.now().isoformat()}\n{cmd}\n", encoding='utf-8')


def run_pipeline(args):
    """Execute the full analysis pipeline."""
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'figures').mkdir(exist_ok=True)
    (output_dir / 'tables').mkdir(exist_ok=True)
    
    report = []
    R = report.append
    
    # ── 1. Load Data ──
    print("="*60)
    print("MaxQuant LC-MS/MS Proteomics Analysis")
    print("="*60)
    
    if args.demo:
        print("\n[DEMO MODE] Using synthetic data...")
        pg_raw, groups, colors = generate_demo_data()
        quant_type = 'iBAQ'
    else:
        print(f"\nLoading from: {args.input}")
        data_dir = Path(args.input).parent
        pg_raw = pd.read_csv(args.input, sep='\t', low_memory=False)
        quant_type = args.quant or 'iBAQ'
        if args.metadata:
            meta = load_metadata(args.metadata)
            print(f"  Metadata: {len(meta)} entries")
        groups, colors = auto_detect_groups(pg_raw)
    
    print(f"  Groups detected: {list(groups.keys())}")
    print(f"  Samples: {sum(len(v) for v in groups.values())}")
    
    # ── 2. Filter ──
    pg = filter_protein_groups(pg_raw)
    pg['description'] = pg.get('Fasta headers', pd.Series()).apply(extract_description)
    
    quant_cols = get_quant_columns(pg, groups, quant_type)
    all_qcols = [c for cols in quant_cols.values() for c in cols]
    
    R(f"# MaxQuant LC-MS/MS Proteomics Report")
    R(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")
    R(f"## 1. Data Summary")
    R(f"- Protein groups (raw): {len(pg_raw)}")
    R(f"- Protein groups (filtered): {len(pg)}")
    R(f"- Groups: {', '.join(groups.keys())}")
    R(f"- Quantification: {quant_type}\n")
    
    # ── 3. Visualizations ──
    print("\nGenerating visualizations...")
    
    plot_protein_counts(pg, quant_cols, colors, output_dir)
    R("![Protein Counts](fig02_proteins_per_group.png)\n")
    
    plot_missing_values(pg, all_qcols, output_dir)
    R("![Missing Values](fig03_missing_values.png)\n")
    
    plot_intensity_distribution(pg, quant_cols, colors, output_dir)
    R("![Distribution](fig04_intensity_distribution.png)\n")
    
    plot_replicate_correlation(pg, quant_cols, colors, output_dir)
    R("![Correlation](fig05_replicate_correlation.png)\n")
    
    plot_venn(pg, quant_cols, colors, output_dir)
    R("![Venn](fig06_venn_diagram.png)\n")
    
    plot_top_proteins(pg, quant_cols, colors, n=20, output_dir=output_dir)
    R("![Top Proteins](fig09_top20_proteins.png)\n")
    
    # ── 4. Differential Abundance ──
    print("Running differential abundance...")
    R("## 2. Differential Abundance\n")
    
    group_names = list(groups.keys())
    comparisons = []
    if args.contrasts:
        for pair in args.contrasts.split(';'):
            a, b = pair.strip().split(',')
            comparisons.append((a.strip(), b.strip()))
    else:
        for i in range(len(group_names)):
            for j in range(i+1, len(group_names)):
                comparisons.append((group_names[i], group_names[j]))
    
    fc_thresh = float(args.fc_threshold or 1.0)
    pval_thresh = float(args.fdr or 0.05)
    
    for idx, (ga, gb) in enumerate(comparisons):
        diff = differential_abundance(pg, quant_cols[ga], quant_cols[gb], ga, gb)
        diff = classify_significance(diff, fc_thresh, pval_thresh)
        n_up = (diff['direction'] == 'Up').sum()
        n_dn = (diff['direction'] == 'Down').sum()
        R(f"### {ga} vs {gb}: {n_up} up, {n_dn} down\n")
        
        plot_volcano(diff, ga, gb, colors, fc_thresh, pval_thresh, output_dir, idx=7+idx)
        R(f"![Volcano {ga} vs {gb}](fig{7+idx:02d}_volcano_{ga}_vs_{gb}.png)\n")
        
        diff.sort_values('pvalue').to_csv(output_dir / 'tables' / f'diff_{ga}_vs_{gb}.csv', index=False)
    
    # ── 5. Allergen Annotation ──
    print("Annotating allergens...")
    R("## 3. Allergen Annotation\n")
    
    keywords = (args.allergen_keywords or ','.join(ALLERGEN_KEYWORDS)).split(',')
    mask = pg['description'].fillna('').str.lower().apply(
        lambda x: any(kw.strip() in x for kw in keywords))
    allergens = pg[mask].copy()
    
    if len(allergens) > 0:
        allergens['allergen_code'] = allergens.apply(
            lambda r: get_allergen_code(r.get('Fasta headers',''), r.get('description','')), axis=1)
        allergens['label'] = allergens.apply(
            lambda r: f"{str(r['description'])[:40]} [{r['allergen_code']}]" if r['allergen_code']
            else str(r['description'])[:50], axis=1)
        allergens.index = allergens['label']
        
        plot_allergen_heatmap(allergens, all_qcols, output_dir)
        R(f"Found {len(allergens)} allergen-related proteins\n")
        R("![Allergen Heatmap](fig10_allergen_heatmap.png)\n")
        allergens.to_csv(output_dir / 'tables' / 'allergen_proteins.csv', index=False)
    else:
        R("No allergen-related proteins found.\n")
    
    # ── 6. Taxonomy ──
    if 'Taxonomy names' in pg.columns:
        print("Taxonomy enrichment...")
        R("## 4. Taxonomy\n")
        tax_cats = pg['Taxonomy names'].fillna('').apply(
            lambda x: categorize_taxonomy(x.split(';')[0].strip()))
        tax_summary = tax_cats.value_counts()
        R("| Category | Count |\n|----------|-------|\n")
        for cat, cnt in tax_summary.items():
            R(f"| {cat} | {cnt} |")
        R("")
        tax_summary.to_csv(output_dir / 'tables' / 'taxonomy_summary.csv')
    
    # ── 7. PCA ──
    print("Running PCA...")
    quant_matrix = pg[all_qcols].replace(0, np.nan).fillna(0)
    if quant_matrix.shape[1] >= 2:
        coords, var_ratio = run_pca(quant_matrix)
        sample_labels = [c.split(' ',1)[-1] for c in all_qcols]
        plot_pca(coords, var_ratio, sample_labels, colors, groups, output_dir)
        R("## 5. PCA\n![PCA](fig12_pca.png)\n")
    
    # ── 8. Modeling (optional) ──
    if args.model and args.model != 'none':
        print(f"Training {args.model} classifier...")
        R(f"## 6. Predictive Modeling ({args.model})\n")
        X = quant_matrix.T.values
        y = []
        for col in all_qcols:
            sample = col.split(' ',1)[-1] if ' ' in col else col
            for g, samps in groups.items():
                if sample in samps:
                    y.append(g); break
        if len(set(y)) >= 2:
            result = train_classifier(X, y, args.model)
            R(f"- CV Accuracy: {result['cv_accuracy_mean']:.3f} ± {result['cv_accuracy_std']:.3f}\n")
            if 'feature_importance' in result:
                top_feat = np.argsort(result['feature_importance'])[-10:]
                R("Top 10 discriminating proteins:\n")
                for fi in reversed(top_feat):
                    R(f"- {pg.iloc[fi].get('description','Unknown')[:50]}: {result['feature_importance'][fi]:.4f}")
                R("")
    
    # ── 9. Save report + reproducibility ──
    print("Writing report...")
    pg.to_csv(output_dir / 'tables' / 'proteinGroups_filtered.csv', index=False)
    (output_dir / 'analysis_report.md').write_text('\n'.join(report), encoding='utf-8')
    write_commands(output_dir, args)
    write_checksums(output_dir)
    
    print(f"\n{'='*60}")
    print(f"Analysis complete!")
    print(f"  Output: {output_dir}")
    print(f"  Report: {output_dir / 'analysis_report.md'}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='MaxQuant LC-MS/MS Proteomics Skill')
    parser.add_argument('--input', help='proteinGroups.txt path')
    parser.add_argument('--input-type', default='maxquant', choices=['maxquant','diann'])
    parser.add_argument('--metadata', help='Sample metadata (SDRF/CSV)')
    parser.add_argument('--quant', default='iBAQ', choices=['iBAQ','lfq','intensity'])
    parser.add_argument('--contrasts', help='Group pairs: "A,B;A,C"')
    parser.add_argument('--fc-threshold', type=float, default=1.0)
    parser.add_argument('--fdr', type=float, default=0.05)
    parser.add_argument('--allergen-keywords', help='Comma-separated keywords')
    parser.add_argument('--model', default='none', choices=['svm','rf','none'])
    parser.add_argument('--output', default='./report')
    parser.add_argument('--demo', action='store_true', help='Run with demo data')
    
    args = parser.parse_args()
    if not args.demo and not args.input:
        parser.error("--input is required unless --demo is used")
    
    run_pipeline(args)


if __name__ == '__main__':
    main()
