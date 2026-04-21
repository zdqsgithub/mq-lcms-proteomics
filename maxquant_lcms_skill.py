"""
MaxQuant LC-MS/MS Proteomics Bioinformatics & Modeling Skill v2
================================================================
Main entry point with mode dispatcher:
  --mode comparison  : Group vs group (default)
  --mode stability   : Time-course degradation analysis
  --mode qc          : QC-only report

Usage:
  python maxquant_lcms_skill.py --demo --output demo_report
  python maxquant_lcms_skill.py --input proteinGroups.txt --mode stability --output report
"""
import sys, os, argparse, hashlib
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from core import (load_maxquant, load_metadata, filter_protein_groups,
                  extract_description, get_quant_columns, log2_transform,
                  impute_missing, compute_qc_metrics, get_allergen_code,
                  categorize_taxonomy, auto_detect_groups)
from stats_engine import (differential_abundance, classify_significance,
                          run_pca, train_classifier, compute_replicate_correlation,
                          benjamini_hochberg, timecourse_analysis)
from visualization import (plot_msms_summary, plot_protein_counts,
                           plot_missing_values, plot_intensity_distribution,
                           plot_replicate_correlation, plot_venn, plot_volcano,
                           plot_allergen_heatmap, plot_pca, plot_top_proteins,
                           plot_timecourse_grid, plot_waterfall,
                           plot_composition_shift, plot_grouped_bar_timecourse)


ALLERGEN_KEYWORDS = [
    'tropomyosin', 'arginine kinase', 'hemocyanin', 'myosin',
    'actin', 'sarcoplasmic calcium', 'aldehyde dehydrogenase',
    'triosephosphate isomerase', 'glyceraldehyde', 'enolase',
    'paramyosin', 'glutathione', 'peroxiredoxin', 'heat shock',
    'superoxide dismutase', 'crustacyanin', 'profilin', 'polcalcin',
    'lipid-transfer protein', 'lipid transfer protein', 'defensin',
    'pectate lyase', 'cysteine proteinase', 'cystatin',
    'galactose oxidase', 'thaumatin', 'chitinase',
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
    demo.loc[:9, 'iBAQ A-1'] *= 10
    demo.loc[:9, 'iBAQ A-2'] *= 10
    groups = {'GroupA': ['A-1', 'A-2'], 'GroupB': ['B-1', 'B-2']}
    colors = {'GroupA': '#4E79A7', 'GroupB': '#E15759'}
    return demo, groups, colors


def write_checksums(output_dir):
    output_dir = Path(output_dir)
    lines = []
    for f in sorted(output_dir.rglob('*')):
        if f.is_file() and f.name != 'checksums.sha256':
            h = hashlib.sha256(f.read_bytes()).hexdigest()
            lines.append(f'{h}  {f.relative_to(output_dir)}')
    (output_dir / 'checksums.sha256').write_text('\n'.join(lines), encoding='utf-8')


def write_commands(output_dir, args):
    cmd = f"python {Path(__file__).name}"
    for k, v in vars(args).items():
        if v is not None and v is not False and k != 'func':
            cmd += f" --{k.replace('_','-')} {v}"
    (Path(output_dir) / 'commands.sh').write_text(
        f"#!/bin/bash\n# Generated {datetime.now().isoformat()}\n{cmd}\n", encoding='utf-8')


def _prepare_data(args):
    """Common data loading and filtering for all modes."""
    if args.demo:
        print("\n[DEMO MODE] Using synthetic data...")
        pg_raw, groups, colors = generate_demo_data()
        quant_type = 'iBAQ'
    else:
        print(f"\nLoading from: {args.input}")
        pg_raw = pd.read_csv(args.input, sep='\t', low_memory=False)
        quant_type = args.quant or 'iBAQ'
        groups, colors = auto_detect_groups(pg_raw, quant_type)

    pg = filter_protein_groups(pg_raw)
    pg['description'] = pg.get('Fasta headers', pd.Series()).apply(extract_description)
    pg['allergen_code'] = pg.apply(
        lambda r: get_allergen_code(r.get('Fasta headers',''), r.get('description','')), axis=1)
    pg['label'] = pg.apply(lambda r:
        f"{r['allergen_code']} - {r['description'][:40]}" if r['allergen_code']
        else r['description'][:50] if r['description'] else str(r.get('Majority protein IDs',''))[:50], axis=1)

    quant_cols = get_quant_columns(pg, groups, quant_type)
    all_qcols = [c for cols in quant_cols.values() for c in cols]

    return pg, pg_raw, groups, colors, quant_type, quant_cols, all_qcols


# ═══════════════════════════════════════════════════════════════
#  MODE: comparison (default — Group A vs B)
# ═══════════════════════════════════════════════════════════════
def run_comparison(args):
    pg, pg_raw, groups, colors, quant_type, quant_cols, all_qcols = _prepare_data(args)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'tables').mkdir(exist_ok=True)

    report = []
    R = report.append
    R(f"# MaxQuant Proteomics Report (Comparison Mode)")
    R(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")
    R(f"- Protein groups: {len(pg_raw)} raw -> {len(pg)} filtered")
    R(f"- Groups: {', '.join(groups.keys())}\n")

    plot_protein_counts(pg, quant_cols, colors, output_dir)
    plot_missing_values(pg, all_qcols, output_dir)
    plot_intensity_distribution(pg, quant_cols, colors, output_dir)
    plot_replicate_correlation(pg, quant_cols, colors, output_dir)
    plot_venn(pg, quant_cols, colors, output_dir)
    plot_top_proteins(pg, quant_cols, colors, n=20, output_dir=output_dir)

    # Differential abundance
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

    R("## Differential Abundance\n")
    for idx, (ga, gb) in enumerate(comparisons):
        diff = differential_abundance(pg, quant_cols[ga], quant_cols[gb], ga, gb)
        diff = classify_significance(diff, fc_thresh, pval_thresh)
        n_up = (diff['direction'] == 'Up').sum()
        n_dn = (diff['direction'] == 'Down').sum()
        R(f"### {ga} vs {gb}: {n_up} up, {n_dn} down\n")
        plot_volcano(diff, ga, gb, colors, fc_thresh, pval_thresh, output_dir, idx=7+idx)
        diff.sort_values('pvalue').to_csv(output_dir / 'tables' / f'diff_{ga}_vs_{gb}.csv', index=False)

    # Allergen annotation
    keywords = (args.allergen_keywords or ','.join(ALLERGEN_KEYWORDS)).split(',')
    mask = pg['description'].fillna('').str.lower().apply(lambda x: any(kw.strip() in x for kw in keywords))
    allergens = pg[mask].copy()
    if len(allergens) > 0:
        allergens.index = allergens['label']
        plot_allergen_heatmap(allergens, all_qcols, output_dir)
        allergens.to_csv(output_dir / 'tables' / 'allergen_proteins.csv', index=False)

    # PCA
    quant_matrix = pg[all_qcols].replace(0, np.nan).fillna(0)
    if quant_matrix.shape[1] >= 2:
        coords, var_ratio = run_pca(quant_matrix)
        sample_labels = [c.split(' ',1)[-1] for c in all_qcols]
        plot_pca(coords, var_ratio, sample_labels, colors, groups, output_dir)

    # Save
    pg.to_csv(output_dir / 'tables' / 'proteinGroups_filtered.csv', index=False)
    (output_dir / 'analysis_report.md').write_text('\n'.join(report), encoding='utf-8')
    write_commands(output_dir, args)
    write_checksums(output_dir)
    print(f"\nComparison analysis complete! -> {output_dir}")


# ═══════════════════════════════════════════════════════════════
#  MODE: stability (time-course degradation)
# ═══════════════════════════════════════════════════════════════
def run_stability(args):
    pg, pg_raw, groups, colors, quant_type, quant_cols, all_qcols = _prepare_data(args)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'tables').mkdir(exist_ok=True)

    report = []
    R = report.append
    group_names = list(groups.keys())

    R(f"# Stability Analysis Report")
    R(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")
    R(f"- Protein groups: {len(pg_raw)} raw -> {len(pg)} filtered")
    R(f"- Time points: {', '.join(group_names)}")
    R(f"- Baseline: {group_names[0]}\n")

    # Heatmap
    heat = pg[all_qcols].replace(0, np.nan).apply(np.log2)
    heat.index = pg['label'].values
    plot_allergen_heatmap(pg.set_index('label'), all_qcols, output_dir)
    R("![Heatmap](fig10_allergen_heatmap.png)\n")

    # Time-course analysis (vectorized)
    print("Running time-course analysis...")
    tc_df = timecourse_analysis(pg, quant_cols, group_names)

    # Visualizations
    print("Generating stability visualizations...")
    plot_timecourse_grid(tc_df, group_names, output_dir)
    R("![Time Course](fig_timecourse_profiles.png)\n")

    last_g = group_names[-1]
    plot_waterfall(tc_df, f'log2FC_{last_g}', f'pct_{last_g}', output_dir,
                   title=f'Stability Ranking ({last_g} vs {group_names[0]})')
    R("![Waterfall](fig_waterfall.png)\n")

    plot_grouped_bar_timecourse(tc_df, group_names, colors, output_dir)
    R("![Grouped Bar](fig_grouped_bar.png)\n")

    plot_composition_shift(tc_df, group_names, output_dir=output_dir)
    R("![Composition](fig_composition.png)\n")

    # Summary table
    R("## Stability Summary\n")
    R("| Protein | Allergen | Baseline iBAQ | Last TP (%) | log2FC | p-value | Trend |")
    R("|---------|----------|---------------|-------------|--------|---------|-------|")
    base_g = group_names[0]
    for _, row in tc_df.sort_values(f'log2FC_{last_g}').iterrows():
        label = str(row.get('label', row.get('description', '')))[:35]
        code = row.get('allergen_code', '') or '-'
        d0 = f"{row[f'mean_{base_g}']:.2e}" if pd.notna(row.get(f'mean_{base_g}')) else '-'
        pct = f"{row[f'pct_{last_g}']:.0f}%" if pd.notna(row.get(f'pct_{last_g}')) else '-'
        fc = f"{row[f'log2FC_{last_g}']:.2f}" if pd.notna(row.get(f'log2FC_{last_g}')) else '-'
        pv = f"{row[f'pval_{last_g}']:.3f}" if pd.notna(row.get(f'pval_{last_g}')) else '-'
        trend = row.get('trend', '-')
        R(f"| {label} | {code} | {d0} | {pct} | {fc} | {pv} | {trend} |")
    R("")

    # Findings
    degrading = tc_df[tc_df['trend']=='Degrading'].sort_values(f'log2FC_{last_g}')
    increasing = tc_df[tc_df['trend']=='Increasing'].sort_values(f'log2FC_{last_g}', ascending=False)
    stable = tc_df[tc_df['trend']=='Stable']

    R(f"## Key Findings\n")
    R(f"### Degrading ({len(degrading)})\n")
    for _, r in degrading.iterrows():
        pct = r.get(f'pct_{last_g}', np.nan)
        pct_str = f"{pct:.0f}%" if pd.notna(pct) else 'N/A'
        R(f"- **{r.get('label','')}**: {pct_str} remaining (log2FC={r.get(f'log2FC_{last_g}',0):.2f})")
    R(f"\n### Stable ({len(stable)})\n")
    for _, r in stable.iterrows():
        R(f"- **{r.get('label','')}**")
    R(f"\n### Increasing ({len(increasing)})\n")
    for _, r in increasing.iterrows():
        pct = r.get(f'pct_{last_g}', np.nan)
        pct_str = f"{pct:.0f}%" if pd.notna(pct) else 'N/A'
        R(f"- **{r.get('label','')}**: {pct_str} (log2FC={r.get(f'log2FC_{last_g}',0):.2f})")
    R("")

    # Save
    tc_df.to_csv(output_dir / 'tables' / 'stability_summary.csv', index=False)
    pg.to_csv(output_dir / 'tables' / 'proteinGroups_filtered.csv', index=False)
    (output_dir / 'stability_report.md').write_text('\n'.join(report), encoding='utf-8')
    write_commands(output_dir, args)
    write_checksums(output_dir)
    print(f"\nStability analysis complete! -> {output_dir}")


# ═══════════════════════════════════════════════════════════════
#  Main dispatcher
# ═══════════════════════════════════════════════════════════════
MODES = {
    'comparison': run_comparison,
    'stability': run_stability,
}


def main():
    parser = argparse.ArgumentParser(description='MaxQuant LC-MS/MS Proteomics Skill v2')
    parser.add_argument('--input', help='proteinGroups.txt path')
    parser.add_argument('--input-type', default='maxquant', choices=['maxquant','diann'])
    parser.add_argument('--metadata', help='Sample metadata (SDRF/CSV)')
    parser.add_argument('--quant', default='iBAQ', choices=['iBAQ','lfq','intensity'])
    parser.add_argument('--mode', default='comparison', choices=list(MODES.keys()),
                        help='Analysis mode: comparison (default) or stability')
    parser.add_argument('--contrasts', help='Group pairs: "A,B;A,C" (comparison mode)')
    parser.add_argument('--fc-threshold', type=float, default=1.0)
    parser.add_argument('--fdr', type=float, default=0.05)
    parser.add_argument('--allergen-keywords', help='Comma-separated keywords')
    parser.add_argument('--model', default='none', choices=['svm','rf','none'])
    parser.add_argument('--output', default='./report')
    parser.add_argument('--demo', action='store_true', help='Run with demo data')

    args = parser.parse_args()
    if not args.demo and not args.input:
        parser.error("--input is required unless --demo is used")

    print("="*60)
    print(f"MaxQuant LC-MS/MS Proteomics Analysis v2 [{args.mode}]")
    print("="*60)

    MODES[args.mode](args)


if __name__ == '__main__':
    main()
