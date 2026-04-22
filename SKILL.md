---
name: maxquant-lcms-proteomics
description: >
  End-to-end MaxQuant LC-MS/MS proteomics bioinformatics skill for allergen extract
  characterization. Covers data loading, QC, filtering, differential abundance analysis,
  taxonomy enrichment, allergen annotation, publication-quality visualization, and
  predictive modeling. Specification-constrained, reproducible, local-first.
version: 2.3.0
metadata:
  openclaw:
    requires:
      bins:
        - python3
      env: []
      config: []
    always: false
    emoji: "🧬"
    homepage: https://github.com/AdvanBio
    os: [darwin, linux, win32]
    install:
      - kind: pip
        package: pandas
      - kind: pip
        package: numpy
      - kind: pip
        package: matplotlib
      - kind: pip
        package: seaborn
      - kind: pip
        package: scipy
      - kind: pip
        package: scikit-learn
      - kind: pip
        package: matplotlib-venn
    trigger_keywords:
      - MaxQuant proteomics analysis
      - LC-MS/MS data processing
      - proteinGroups.txt analysis
      - allergen proteomics
      - label-free quantification analysis
      - iBAQ differential abundance
      - proteomics bioinformatics pipeline
---

# MaxQuant LC-MS/MS Proteomics Bioinformatics & Modeling Skill

> A comprehensive, specification-constrained agent skill for processing MaxQuant
> output files. Integrates bioinformatics interpretation (inspired by ClawBio),
> publication-quality visualization (inspired by K-Dense-AI Scientific Agent Skills),
> structured development methodology (inspired by Superpowers), and autonomous
> model optimization (inspired by Autoresearch).

---

## Skill Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MaxQuant LCMS Proteomics Skill                   │
├─────────────┬──────────────┬──────────────┬────────────────────────┤
│  Module 1   │   Module 2   │   Module 3   │      Module 4          │
│  Data QC &  │ Differential │ Visualization│  Modeling &            │
│  Filtering  │  Abundance   │  & Reporting │  Optimization          │
│             │  & Taxonomy  │              │                        │
│ (MaxQuant   │ (ClawBio     │ (K-Dense-AI  │ (Autoresearch          │
│  Tutorial)  │  inspired)   │  inspired)   │  inspired)             │
├─────────────┴──────────────┴──────────────┴────────────────────────┤
│              Superpowers Development Methodology                    │
│  brainstorming → writing-plans → TDD → code-review → finish       │
├────────────────────────────────────────────────────────────────────┤
│              Reproducibility Bundle (commands.sh, checksums)        │
└────────────────────────────────────────────────────────────────────┘
```

---

## Module 1: Data Loading, QC & Filtering

### What It Does
- Loads MaxQuant output files: `proteinGroups.txt`, `peptides.txt`, `evidence.txt`, `summary.txt`, `msmsScans.txt`, `parameters.txt`
- Also supports DIA-NN output (`.tsv` / `.txt`)
- Parses experimental design from SDRF/metadata files
- Filters: Reverse hits, Potential contaminants, Only-identified-by-site
- Extracts protein descriptions from UniProt-style FASTA headers
- Computes per-sample QC metrics: MS/MS submitted, identified, ID rate, peptide counts

### MaxQuant Output File Reference

| File | Description | Key Columns |
|------|-------------|-------------|
| `proteinGroups.txt` | Protein-level quantification | Protein IDs, LFQ intensity, iBAQ, Fasta headers, Taxonomy |
| `peptides.txt` | Peptide-level data | Sequence, Intensity, Missed cleavages, Score |
| `evidence.txt` | PSM-level evidence | Modified sequence, Charge, m/z, Retention time, Match type |
| `msmsScans.txt` | MS/MS scan metadata | Raw file, Scan number, Retention time, Ion injection time |
| `summary.txt` | Run-level summary statistics | MS/MS submitted, identified, Peptide sequences |
| `parameters.txt` | Search parameters used | Enzyme, Variable modifications, Fixed modifications |

### Filtering Logic
```python
# Standard MaxQuant filtering (never skip these)
for col in ['Reverse', 'Potential contaminant', 'Only identified by site']:
    mask &= df[col].fillna('').str.strip() != '+'
```

### FASTA Header Parsing
- **Identifier parse rule**: `>.*\|(.*)\|` extracts UniProt accession
- **Description parse rule**: `>(.*) OS` extracts protein name
- Supports both `sp|` (Swiss-Prot) and `tr|` (TrEMBL) prefixes

### Quantification Strategies
| Strategy | Column Pattern | Use Case |
|----------|---------------|----------|
| LFQ | `LFQ intensity <sample>` | Normalized label-free quantification |
| iBAQ | `iBAQ <sample>` | Intensity-Based Absolute Quantification |
| Raw Intensity | `Intensity <sample>` | Unnormalized signal |

---

## Module 2: Differential Abundance & Taxonomy Enrichment

### Differential Abundance Analysis
- **Transformation**: log2 scaling of intensities (LFQ or iBAQ)
- **Missing value handling**: Down-shifted Gaussian imputation
  - Default: `shift = 1.8`, `scale = 0.3`
  - Assumption: missing = low-abundance (MNAR)
- **Statistical testing**: Welch's t-test (unequal variance)
- **Multiple testing**: Benjamini-Hochberg FDR correction
- **s0-based thresholding** (Giai Gianetto et al. 2016): Combines fold-change and p-value for stable significance calls
- **Significance thresholds**: |log2FC| > 1.0, FDR < 0.05

### Taxonomy Enrichment
- Parses `Taxonomy names` column from proteinGroups
- Categorizes into biological groups:
  - Shrimp/Crustacean (Penaeus, Macrobrachium, etc.)
  - Dust Mite (Dermatophagoides)
  - Bacteria (Vibrio, Bacillus, etc.)
  - Other Arthropod
- Computes per-group taxonomic composition
- Generates species-level abundance profiles per sample group

### Allergen Annotation
- Maps proteins to WHO/IUIS allergen nomenclature
- Extracts allergen codes from FASTA headers (e.g., `Pen a 1`, `Der p 2`)
- Keyword-based fallback mapping for:
  - Tropomyosin → Group 1
  - Arginine kinase → Group 2
  - Myosin light chain → Group 3
  - Sarcoplasmic calcium-binding → Group 4
  - Paramyosin → Group 11
  - Hemocyanin, Enolase, TPI, GAPDH, etc.

---

## Module 3: Visualization & Reporting

### Publication-Quality Figures (K-Dense-AI inspired)
All figures are generated at 150+ DPI with consistent styling:

| Figure | Type | Description |
|--------|------|-------------|
| MS/MS Summary | Bar chart | Submitted vs. identified spectra per sample |
| Protein Counts | Bar chart | Protein groups detected per group (iBAQ > 0) |
| Missing Values | Heatmap | Missing value pattern across samples |
| Intensity Distribution | Histogram | log2(iBAQ) density per sample |
| Replicate Correlation | Scatter | log2 iBAQ Rep1 vs Rep2 with Pearson r |
| Venn Diagram | Venn | Protein group overlap between groups |
| Volcano Plots | Scatter | log2FC vs -log10(p) with significance coloring |
| Taxonomy Distribution | Horizontal bar | Species assignments across protein groups |
| Top N Abundance | Horizontal bar | Most abundant proteins per group |
| Allergen Heatmap | Clustered heatmap | Key allergen proteins across all samples |
| Top 50 Heatmap | Heatmap | Overall most abundant 50 proteins |
| PCA Plot | Scatter | Principal component analysis of samples |

### Styling Standards
```python
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.facecolor': 'white',
})
sns.set_style("whitegrid")
```

### Color Palettes
- Group-specific colors: curated, high-contrast palette
- Heatmaps: `YlOrRd` for allergens, `viridis` for general
- Significance: red/blue for up/down, grey for non-significant

### Report Output
- Comprehensive Markdown report (`analysis_report.md`)
- All figures embedded with relative paths
- Summary tables in Markdown format
- CSV exports for downstream analysis

---

## Module 4: Modeling & Optimization (Autoresearch-inspired)

### Autonomous Experimental Loop
Inspired by Karpathy's autoresearch pattern:
1. **Baseline**: Run standard analysis pipeline
2. **Hypothesis**: Agent proposes parameter/method modification
3. **Experiment**: Execute modified pipeline (time-budgeted)
4. **Evaluate**: Compare metrics (protein count, FDR, correlation)
5. **Decision**: Keep improvement or discard, log result
6. **Iterate**: Repeat with next hypothesis

### Optimizable Parameters
| Parameter | Range | Metric |
|-----------|-------|--------|
| Imputation shift | 1.0-3.0 | Downstream DE sensitivity |
| Imputation scale | 0.1-0.5 | Imputation distribution fit |
| FC threshold | 0.5-2.0 | Significant hit count |
| FDR threshold | 0.01-0.10 | False positive control |
| Min unique peptides | 1-3 | Protein confidence |
| Normalization method | median/quantile/none | Cross-sample comparability |

### Predictive Modeling
- **Sample classification**: SVM/Random Forest on protein profiles
- **Biomarker discovery**: Feature importance from ensemble models
- **Cross-validation**: Leave-one-out or k-fold for small sample sizes
- **Model evaluation**: AUC, accuracy, confusion matrix

---

## Development Methodology (Superpowers-inspired)

### Workflow
1. **Brainstorming**: Understand experimental design, organism, allergens of interest
2. **Planning**: Break analysis into tasks with verification steps
3. **TDD**: Write test assertions first, then implement
4. **Code Review**: Verify against plan, check statistical validity
5. **Finish**: Generate reproducibility bundle

### Test-Driven Development
Every analysis step has testable assertions:
```python
# Example: filtering must remove reverse hits
assert filtered_df[filtered_df['Reverse'] == '+'].empty
# Example: log2 transform must not contain infinities
assert not np.isinf(log2_data).any().any()
# Example: volcano plot must have correct number of significant hits
assert len(sig_up) + len(sig_down) == expected_sig_count
```

---

## Reproducibility Bundle (ClawBio-inspired)

Every analysis run produces:
```
report/
├── analysis_report.md          # Full report with embedded figures
├── figures/                    # Publication-quality PNGs (150+ DPI)
│   ├── fig01_msms_summary.png
│   ├── fig02_proteins_per_group.png
│   ├── ...
│   └── fig12_pca_plot.png
├── tables/                     # CSV data exports
│   ├── proteinGroups_filtered.csv
│   ├── allergen_proteins.csv
│   ├── diff_GroupA_vs_GroupB.csv
│   ├── taxonomy_summary.csv
│   └── model_results.csv
├── commands.sh                 # Exact commands to reproduce
├── environment.yml             # Conda/pip environment snapshot
└── checksums.sha256            # SHA-256 of every input and output
```

---

## Constraints & Guardrails

### This Skill DOES:
- ✅ Process MaxQuant and DIA-NN output files
- ✅ Perform differential abundance analysis with proper statistics
- ✅ Generate publication-quality visualizations
- ✅ Annotate proteins with allergen nomenclature and taxonomy
- ✅ Build predictive models for sample classification
- ✅ Produce reproducible analysis bundles
- ✅ Handle multi-group comparisons (>2 groups)
- ✅ Support both LFQ and iBAQ quantification

### This Skill DOES NOT:
- ❌ Process raw mass spectrometry data (.raw, .mzML)
- ❌ Perform peptide identification or database search
- ❌ Replace MaxQuant or DIA-NN search engines
- ❌ Provide clinical diagnostic recommendations
- ❌ Upload data to any cloud service

### Statistical Cautions:
- All trend classifications (Degrading/Stable/Increasing) require **both** |log2FC| > 0.5 **and** p < 0.05
- Stability summary tables are filtered to p < 0.05 only
- Correlation outputs show "insufficient data" when sample sizes preclude valid testing
- t-test reliability depends on sufficient replicates (>=3 per group recommended)
- Imputation assumes MNAR (missing not at random) -- may not hold for all designs
- Multiple testing correction is essential for proteome-wide comparisons
- With only 2 replicates, p-values have very limited statistical power

### Deep-Stability Pipeline (10 Steps)

The `--mode deep-stability` pipeline runs 10 analysis steps:

| Step | Analysis | Output |
|:---:|----------|--------|
| 1 | Functional Enrichment | Trend distribution by functional category |
| 2 | MW Distribution | Molecular weight by trend |
| 3 | Methionine Oxidation | Oxidation ratio kinetics, correlation with degradation |
| 4 | Deamidation (NQ) Sites | Deamidation site tracking, correlation |
| 5 | Protease Activity | Semi-tryptic kinetics, endogenous protease inventory |
| 6 | Coverage Kinetics | Unfolding vs aggregation signature |
| 7 | Sequence Composition | GRAVY, Pro, hydrophobic content by trend |
| 8 | Degradation Routes | 4-panel overview (semi-tryptic, peptides, acetylation, MC) |
| 9 | Fragment Profiling | P1 cleavage specificity, calpain/caspase classification |
| 10 | **Biophysical Analysis** | UniProt sequence fetch, pI, aliphatic index, aggregation score |

Step 10 requires internet connectivity to fetch sequences from UniProt REST API.
It exports a FASTA file for external Tm prediction tools (DeepSTABp, TANGO, CamSol).

### Quantification Fallback
- Pipeline automatically falls back: iBAQ -> LFQ -> Intensity
- Reports which quantification was used in the header

### No Hallucinated Science:
- All methods based on established proteomics workflows
- Allergen codes follow WHO/IUIS nomenclature
- Statistical methods cite published references
- Agent must NOT improvise bioinformatics decisions from training data

---

## Input Formats

### Required
1. MaxQuant `proteinGroups.txt` — protein-level quantification

### Optional (enhance analysis)
2. `peptides.txt` — peptide-level data
3. `evidence.txt` — PSM-level evidence with match-between-runs info
4. `summary.txt` — run-level QC statistics
5. `msmsScans.txt` — MS/MS scan metadata
6. `parameters.txt` — MaxQuant search parameters
7. `sdrf.tsv` or `metadata.csv` — experimental design with sample→group mapping

### Metadata Requirements
- Tab-separated or comma-separated
- Must include: `sample_id` (matching raw file names) and `group` columns
- Optional: `replicate`, `batch`, `condition`, `organism`

---

## Usage

### Demo (with built-in test data)
```bash
python maxquant_lcms_skill.py --demo --output report_dir
```

### Standard Analysis
```bash
python maxquant_lcms_skill.py \
  --input proteinGroups.txt \
  --metadata sdrf.tsv \
  --quant iBAQ \
  --contrasts "GroupA,GroupB;GroupA,GroupC" \
  --output report_dir
```

### Full Pipeline with Modeling
```bash
python maxquant_lcms_skill.py \
  --input proteinGroups.txt \
  --peptides peptides.txt \
  --evidence evidence.txt \
  --summary summary.txt \
  --metadata sdrf.tsv \
  --quant iBAQ \
  --contrasts "Greer,Inhouse;Greer,Phadia;Inhouse,Phadia" \
  --allergen-keywords "tropomyosin,arginine kinase,hemocyanin" \
  --model svm \
  --output report_dir
```

### Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input` | proteinGroups.txt path | required |
| `--input-type` | `maxquant` or `diann` | maxquant |
| `--metadata` | Sample metadata file | auto-detect |
| `--quant` | Quantification: `lfq`, `ibaq`, `intensity` | ibaq |
| `--contrasts` | Semicolon-separated group pairs | all pairwise |
| `--fc-threshold` | log2FC cutoff | 1.0 |
| `--fdr` | FDR threshold | 0.05 |
| `--s0` | s0 parameter for variance stabilization | 0.1 |
| `--imputation-shift` | Gaussian imputation shift | 1.8 |
| `--imputation-scale` | Gaussian imputation scale | 0.3 |
| `--allergen-keywords` | Comma-separated allergen search terms | built-in list |
| `--model` | Prediction model: `svm`, `rf`, `none` | none |
| `--output` | Output directory | ./report |

---

## References

1. Cox J, Mann M. MaxQuant enables high peptide identification rates. Nat Biotechnol. 2008;26(12):1367-72.
2. Tyanova S, Temu T, Cox J. The MaxQuant computational platform for mass spectrometry-based shotgun proteomics. Nat Protoc. 2016;11(12):2301-19.
3. Giai Gianetto Q, et al. Uses and misuses of the fudge factor in quantitative discovery proteomics. Proteomics. 2016;16(14):1955-60.
4. Galaxy Training Network. Label-free data analysis using MaxQuant. GTN Tutorial GTN:T00218.
5. ClawBio. Specification-constrained bioinformatics agent skills. https://github.com/ClawBio/ClawBio
6. K-Dense-AI. Scientific Agent Skills. https://github.com/K-Dense-AI/scientific-agent-skills
7. obra/superpowers. Agentic skills framework & development methodology. https://github.com/obra/superpowers
8. karpathy/autoresearch. AI agents running research automatically. https://github.com/karpathy/autoresearch
