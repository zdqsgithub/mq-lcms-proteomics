"""
MaxQuant LC-MS/MS Proteomics Bioinformatics & Modeling Skill v2.2
=================================================================
Main entry point with mode dispatcher:
  --mode comparison     : Group vs group (default)
  --mode stability      : Time-course degradation analysis
  --mode deep-stability : Stability + pathway + oxidation + protease

Usage:
  python maxquant_lcms_skill.py --demo --output demo_report
  python maxquant_lcms_skill.py --input proteinGroups.txt --mode stability --output report
  python maxquant_lcms_skill.py --input-dir ./txt --mode deep-stability --output report
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
                           plot_composition_shift, plot_grouped_bar_timecourse,
                           plot_functional_enrichment, plot_mw_by_trend,
                           plot_oxidation_heatmap, plot_degradation_routes_summary)
from degradation_routes import (functional_enrichment, analyze_oxidation_sites,
                                correlate_oxidation_degradation,
                                semi_tryptic_kinetics, inventory_proteases_phosphatases,
                                peptide_appearance, count_deamidation_motifs,
                                detect_semi_tryptic, coverage_kinetics,
                                analyze_deamidation_sites, sequence_composition)


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
#  MODE: deep-stability (stability + pathway + oxidation + protease)
# ═══════════════════════════════════════════════════════════════
def run_deep_stability(args):
    """Full stability analysis with degradation route characterization."""
    # First run standard stability
    run_stability(args)

    output_dir = Path(args.output)
    data_dir = Path(args.input_dir) if args.input_dir else Path(args.input).parent
    group_names = None  # Will be re-detected

    # Reload stability results
    stab_csv = output_dir / 'tables' / 'stability_summary.csv'
    pg_csv = output_dir / 'tables' / 'proteinGroups_filtered.csv'
    if not stab_csv.exists() or not pg_csv.exists():
        print("Stability results not found, skipping deep analysis")
        return

    stab_df = pd.read_csv(stab_csv)
    pg = pd.read_csv(pg_csv)
    _, _, groups, colors, _, _, _ = _prepare_data(args)
    group_names = list(groups.keys())

    report = ["\n## Deep Stability Analysis\n"]
    R = report.append

    # 1. Functional enrichment
    print("Deep: Functional enrichment...")
    cat_counts, cat_pcts, _ = functional_enrichment(stab_df)
    plot_functional_enrichment(cat_pcts, cat_counts, output_dir)
    R("### Functional Enrichment\n")
    R("![Enrichment](fig_functional_enrichment.png)\n")

    # 2. MW analysis
    if 'Mol. weight [kDa]' in stab_df.columns:
        plot_mw_by_trend(stab_df, output_dir)
        R("### Molecular Weight\n")
        R("![MW](fig_mw_by_trend.png)\n")

    # 3. Oxidation analysis
    ox_path = data_dir / 'Oxidation (M)Sites.txt'
    if ox_path.exists():
        print("Deep: Oxidation analysis...")
        from core import extract_description
        ox_df = analyze_oxidation_sites(ox_path, groups, extract_description)
        plot_oxidation_heatmap(ox_df, group_names, output_dir)
        R("### Oxidation Kinetics\n")
        R("![Oxidation](fig_oxidation_heatmap.png)\n")

        merged, r_val, p_val = correlate_oxidation_degradation(ox_df, stab_df, group_names)
        if r_val is not None:
            R(f"- Oxidation vs degradation: r={r_val:.3f}, p={p_val:.3f}\n")
        ox_df.to_csv(output_dir / 'tables' / 'oxidation_sites.csv', index=False)

    # 4. Deamidation site analysis
    deam_path = data_dir / 'Deamidation (NQ)Sites.txt'
    if deam_path.exists():
        print("Deep: Deamidation analysis...")
        from core import extract_description
        deam_df = analyze_deamidation_sites(deam_path, groups, extract_description)
        R(f"### Deamidation (NQ) Sites\n")
        R(f"- Total sites: **{len(deam_df)}**")
        last = group_names[-1]
        if 'ratio_change' in deam_df.columns:
            n_inc = (deam_df['ratio_change'] > 0).sum()
            R(f"- Increasing deamidation (D7 > D0): **{n_inc}**")
        merged_d, r_d, p_d = correlate_oxidation_degradation(deam_df, stab_df, group_names)
        if r_d is not None:
            R(f"- Deamidation vs degradation: r={r_d:.3f}, p={p_d:.3f}")
        deam_df.to_csv(output_dir / 'tables' / 'deamidation_sites.csv', index=False)
        R("")

    # 5. Protease & peptide analysis
    pep_path = data_dir / 'peptides.txt'
    ev_path = data_dir / 'evidence.txt'
    if pep_path.exists():
        print("Deep: Protease analysis...")
        pep_df = pd.read_csv(pep_path, sep='\t', low_memory=False)
        for col in ['Reverse', 'Potential contaminant']:
            if col in pep_df.columns:
                pep_df = pep_df[pep_df[col].fillna('').str.strip() != '+']

        semi_ratios, pep_annotated = semi_tryptic_kinetics(pep_df, groups)
        pep_info = peptide_appearance(pep_df, groups)
        n_deamid = count_deamidation_motifs(pep_df)

        R("### Protease Activity\n")
        R(f"- New peptides at last TP: **{pep_info['gained_at_last']}**")
        R(f"- Lost peptides: **{pep_info['lost_from_baseline']}**")
        R(f"- Deamidation-prone motifs: **{n_deamid}**\n")

        proteases, phosphatases = inventory_proteases_phosphatases(stab_df)
        if len(proteases) > 0:
            R("### Proteases Detected\n")
            R("| Protein | Trend | Risk |")
            R("|---------|-------|------|")
            for _, r in proteases.iterrows():
                R(f"| {str(r.get('description',''))[:40]} | {r.get('trend','-')} | {r.get('risk','-')} |")
            R("")

        # 6. Coverage kinetics (unfolding evidence)
        print("Deep: Coverage kinetics...")
        acc_map = {}
        for _, r in stab_df.iterrows():
            for acc in str(r.get('Majority protein IDs', '')).split(';'):
                acc_map[acc.strip()] = {'trend': r.get('trend', 'Unknown'),
                                        'description': r.get('description', '')}
        cov_df = coverage_kinetics(pep_df, groups, acc_map)
        R("### Coverage Kinetics (Unfolding Evidence)\n")
        for trend in ['Degrading', 'Stable', 'Increasing']:
            sub = cov_df[cov_df['trend'] == trend]
            if len(sub) > 0:
                R(f"- **{trend}**: mean peptide change = {sub['pep_change'].mean():+.1f} ({sub['pep_pct_change'].mean():+.1f}%)")
        R("")
        cov_df.to_csv(output_dir / 'tables' / 'coverage_kinetics.csv')

        # 7. Sequence composition
        print("Deep: Sequence composition...")
        comp_df = sequence_composition(pep_df, acc_map)
        R("### Sequence Features Predicting Stability\n")
        from scipy import stats as sp_stats
        for feat in ['GRAVY', 'pct_Pro', 'pct_hydrophobic']:
            d_vals = comp_df[comp_df['trend']=='Degrading'][feat].dropna()
            i_vals = comp_df[comp_df['trend']=='Increasing'][feat].dropna()
            if len(d_vals) > 3 and len(i_vals) > 3:
                try:
                    _, p = sp_stats.mannwhitneyu(d_vals.astype(float), i_vals.astype(float))
                    sig = ' **' if p < 0.05 else ''
                    R(f"- {feat}: Degrading={d_vals.mean():.2f} vs Increasing={i_vals.mean():.2f} (p={p:.3f}{sig})")
                except Exception:
                    pass
        R("")
        comp_df.to_csv(output_dir / 'tables' / 'sequence_composition.csv')

        # Compute missed cleavages and acetylation
        mc_means = {}
        acetyl_ratios = {}
        if ev_path.exists():
            ev = pd.read_csv(ev_path, sep='\t', low_memory=False)
            for col in ['Reverse', 'Potential contaminant']:
                if col in ev.columns:
                    ev = ev[ev[col].fillna('').str.strip() != '+']
            for g, samps in groups.items():
                sub = ev[ev['Experiment'].isin(samps)]
                mc_means[g] = sub['Missed cleavages'].mean() if len(sub) > 0 else 0
                ace = sub['Modifications'].fillna('').str.contains('Acetyl', case=False).sum()
                acetyl_ratios[g] = ace / len(sub) * 100 if len(sub) > 0 else 0

        pep_counts = pep_info.get('present_per_group', {})
        plot_degradation_routes_summary(semi_ratios, pep_counts, acetyl_ratios,
                                        mc_means, group_names, colors, output_dir)
        R("### Degradation Routes Summary\n")
        R("![Routes](fig_degradation_routes.png)\n")

    # Append to existing report
    existing = (output_dir / 'stability_report.md').read_text(encoding='utf-8')
    (output_dir / 'stability_report.md').write_text(
        existing + '\n'.join(report), encoding='utf-8')
    write_checksums(output_dir)
    print(f"\nDeep stability analysis complete! -> {output_dir}")


# ═══════════════════════════════════════════════════════════════
#  Main dispatcher
# ═══════════════════════════════════════════════════════════════
MODES = {
    'comparison': run_comparison,
    'stability': run_stability,
    'deep-stability': run_deep_stability,
}


def main():
    parser = argparse.ArgumentParser(description='MaxQuant LC-MS/MS Proteomics Skill v2.1')
    parser.add_argument('--input', help='proteinGroups.txt path')
    parser.add_argument('--input-dir', help='MaxQuant txt/ directory (for deep-stability)')
    parser.add_argument('--input-type', default='maxquant', choices=['maxquant','diann'])
    parser.add_argument('--metadata', help='Sample metadata (SDRF/CSV)')
    parser.add_argument('--quant', default='iBAQ', choices=['iBAQ','lfq','intensity'])
    parser.add_argument('--mode', default='comparison', choices=list(MODES.keys()),
                        help='Analysis mode: comparison, stability, or deep-stability')
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

    # Auto-set input-dir from input path
    if not args.input_dir and args.input:
        args.input_dir = str(Path(args.input).parent)

    print("="*60)
    print(f"MaxQuant LC-MS/MS Proteomics Analysis v2.1 [{args.mode}]")
    print("="*60)

    MODES[args.mode](args)


if __name__ == '__main__':
    main()
