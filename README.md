# MaxQuant LC-MS/MS Proteomics Bioinformatics & Modeling Skill

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-50%2F50%20passed-brightgreen.svg)](#test-suite)

A comprehensive, specification-constrained **agent skill** for end-to-end processing of MaxQuant LC-MS/MS proteomics data. Designed for allergen extract characterization, but applicable to any label-free quantitative proteomics workflow.

This skill integrates methodologies from five established open-source projects into a unified, testable, and reproducible bioinformatics library:

| Source | What It Provides |
|--------|-----------------|
| [Galaxy MaxQuant Tutorial](https://training.galaxyproject.org/training-material/topics/proteomics/tutorials/maxquant-label-free/tutorial.html) | Pipeline logic — filtering rules, FASTA header parsing, QC metrics |
| [ClawBio](https://github.com/ClawBio/ClawBio) | Skill specification — declarative `SKILL.md` contract, reproducibility bundles |
| [K-Dense-AI Scientific Agent Skills](https://github.com/K-Dense-AI/scientific-agent-skills) | Visualization — publication-quality figure library, styling standards |
| [Superpowers](https://github.com/obra/superpowers) | Development methodology — TDD, spec-first design, brainstorming workflow |
| [Autoresearch](https://github.com/karpathy/autoresearch) | Modeling — SVM/RF classifiers, iterative optimization loop |

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
  - [Demo Mode](#demo-mode)
  - [Standard Analysis](#standard-analysis)
  - [Full Pipeline with Modeling](#full-pipeline-with-modeling)
- [Architecture](#architecture)
- [Modules](#modules)
  - [core.py — Data Engine](#corepy--data-engine)
  - [stats_engine.py — Statistical Analysis](#stats_enginepy--statistical-analysis)
  - [visualization.py — Publication-Quality Figures](#visualizationpy--publication-quality-figures)
- [CLI Reference](#cli-reference)
- [Supported Input Formats](#supported-input-formats)
- [Output Structure](#output-structure)
- [Test Suite](#test-suite)
- [Skill Specification (SKILL.md)](#skill-specification-skillmd)
- [Constraints & Guardrails](#constraints--guardrails)
- [References](#references)
- [License](#license)

---

## Features

- **Data Loading & QC**: Load MaxQuant `proteinGroups.txt` (and optional `peptides.txt`, `evidence.txt`, `summary.txt`), auto-detect sample groups, compute per-sample QC metrics
- **Standard Filtering**: Remove reverse hits, potential contaminants, and only-identified-by-site entries following the Galaxy Training Network best practices
- **Flexible Quantification**: Support for iBAQ, LFQ, and raw intensity columns
- **Differential Abundance**: Welch's t-test with Benjamini-Hochberg FDR correction and s0-based variance stabilization
- **Missing Value Imputation**: Down-shifted Gaussian imputation (shift=1.8, scale=0.3) for MNAR data
- **Allergen Annotation**: Automatic mapping to WHO/IUIS allergen nomenclature from FASTA headers
- **Taxonomy Enrichment**: Species-level categorization (shrimp, dust mite, bacteria, etc.)
- **12 Visualization Types**: Volcano, heatmap, PCA, Venn, correlation, bar charts, and more
- **Predictive Modeling**: SVM and Random Forest classifiers with leave-one-out cross-validation
- **Reproducibility Bundle**: Every run generates `commands.sh`, `environment.yml`, and `checksums.sha256`
- **Local-First**: All processing runs locally. No data is uploaded to any cloud service.

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/zdqsgithub/mq-lcms-proteomics.git
cd mq-lcms-proteomics

# Install dependencies
pip install -r requirements.txt

# Run the demo (no data files needed)
python maxquant_lcms_skill.py --demo --output demo_report

# Run the test suite
python test_skill.py
```

---

## Installation

### Requirements

- Python 3.10+
- Dependencies:

```
pandas>=1.5
numpy>=1.23
matplotlib>=3.6
seaborn>=0.12
scipy>=1.10
scikit-learn>=1.2
matplotlib-venn>=0.11
```

### Install

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn matplotlib-venn
```

Or use the provided requirements file:

```bash
pip install -r requirements.txt
```

---

## Usage

### Demo Mode

Run the full pipeline with synthetic data to verify installation:

```bash
python maxquant_lcms_skill.py --demo --output demo_report
```

This generates a complete report with 9+ figures, CSV tables, and a reproducibility bundle using built-in demo data (100 synthetic protein groups, 2 groups, 2 replicates each).

### Standard Analysis

Point the skill at your MaxQuant `proteinGroups.txt`:

```bash
python maxquant_lcms_skill.py \
  --input /path/to/proteinGroups.txt \
  --quant iBAQ \
  --output my_report
```

The skill auto-detects sample groups from column names (e.g., `iBAQ GroupA-1`, `iBAQ GroupA-2` → group `GroupA`).

### Full Pipeline with Modeling

```bash
python maxquant_lcms_skill.py \
  --input proteinGroups.txt \
  --metadata sdrf.tsv \
  --quant iBAQ \
  --contrasts "Greer,Inhouse;Greer,Phadia;Inhouse,Phadia" \
  --allergen-keywords "tropomyosin,arginine kinase,hemocyanin,myosin" \
  --model rf \
  --fc-threshold 1.0 \
  --fdr 0.05 \
  --output full_report
```

### Specifying Contrasts

By default, the skill performs all pairwise comparisons. To specify exact contrasts:

```bash
--contrasts "GroupA,GroupB;GroupA,GroupC"
```

Each pair is separated by `;`, with group names separated by `,`.

---

## Architecture

```
+---------------------------------------------------------------------+
|                    MaxQuant LCMS Proteomics Skill                     |
+-------------+--------------+--------------+--------------------------+
|  Module 1   |   Module 2   |   Module 3   |      Module 4            |
|  Data QC &  | Differential |  Visualiza-  |  Modeling &              |
|  Filtering  |  Abundance   |  tion &      |  Optimization            |
|             |  & Taxonomy  |  Reporting   |                          |
|  (Galaxy    |  (ClawBio    |  (K-Dense-AI |  (Autoresearch           |
|   Tutorial) |   inspired)  |   inspired)  |   inspired)              |
+-------------+--------------+--------------+--------------------------+
|              Superpowers Development Methodology                      |
|   brainstorming -> writing-plans -> TDD -> code-review -> finish     |
+----------------------------------------------------------------------+
|              Reproducibility Bundle (commands.sh, checksums.sha256)   |
+----------------------------------------------------------------------+
```

### Data Flow

```
proteinGroups.txt
        |
        v
  [1] Load & Filter  ----> QC metrics, missing value heatmap
        |
        v
  [2] Transform       ----> log2(iBAQ), imputation
        |
        v
  [3] Statistics       ----> Welch's t-test, BH-FDR, volcano plots
        |
        v
  [4] Annotation       ----> Allergen codes (WHO/IUIS), taxonomy
        |
        v
  [5] Visualization    ----> 12 figure types (PNG, 150 DPI)
        |
        v
  [6] Modeling         ----> SVM/RF classifier, feature importance
        |
        v
  [7] Report           ----> Markdown report + CSV + checksums
```

---

## Modules

### `core.py` — Data Engine

The foundational module for all data operations.

| Function | Description |
|----------|-------------|
| `load_maxquant(data_dir)` | Load all MaxQuant output files from a directory |
| `load_metadata(meta_path)` | Load experimental metadata (SDRF/CSV/TSV) |
| `filter_protein_groups(df)` | Remove reverse, contaminant, site-only entries |
| `filter_peptides(df)` | Remove reverse and contaminant peptides |
| `extract_description(fasta_header)` | Parse protein name from UniProt FASTA headers |
| `get_quant_columns(df, groups, quant_type)` | Get column names for iBAQ/LFQ/intensity |
| `log2_transform(df, columns)` | Log2 transform with zero→NaN replacement |
| `impute_missing(df, shift, scale)` | Down-shifted Gaussian imputation for MNAR data |
| `compute_qc_metrics(summary_df, groups)` | Extract per-sample QC from summary.txt |
| `get_allergen_code(fasta_header, description)` | Map protein to WHO/IUIS allergen code |
| `categorize_taxonomy(tax_name)` | Categorize species into biological groups |

**Filtering Logic:**
```python
# Standard MaxQuant filtering (Galaxy Training Network best practice)
for col in ['Reverse', 'Potential contaminant', 'Only identified by site']:
    mask &= df[col].fillna('').str.strip() != '+'
```

**FASTA Header Parsing:**
```python
# Input:  "sp|P02768|ALBU_HUMAN Serum albumin OS=Homo sapiens"
# Output: "Serum albumin"
extract_description(fasta_header)
```

**Allergen Code Mapping:**
```python
# Direct extraction from FASTA: "Tropomyosin Pen a 1.0102" → "Pen a 1"
# Keyword fallback:             "tropomyosin" + organism PENVA → "Pen v 1"
get_allergen_code(fasta_header, description)
```

### `stats_engine.py` — Statistical Analysis

All statistical methods for proteomics data analysis and predictive modeling.

| Function | Description |
|----------|-------------|
| `differential_abundance(df, cols_a, cols_b)` | Welch's t-test between two groups |
| `benjamini_hochberg(pvalues, fdr)` | FDR correction for multiple testing |
| `s0_significance(log2fc, pvalue, s0, fdr)` | s0-based variance-stabilized thresholding |
| `classify_significance(diff_df, fc_thresh, pval_thresh)` | Classify Up/Down/NS |
| `run_pca(data_matrix, n_components)` | PCA dimensionality reduction |
| `train_classifier(X, y, model_type, cv)` | SVM/RF with cross-validation |
| `compute_replicate_correlation(df, cols)` | Pearson correlation between replicates |

**Differential Abundance:**
```python
# Computes per-protein: log2FC, p-value, -log10(p)
result = differential_abundance(pg, cols_groupA, cols_groupB, 'GroupA', 'GroupB')
```

**Predictive Modeling:**
```python
# Train Random Forest with leave-one-out CV
result = train_classifier(X, y, model_type='rf', cv='loo')
print(f"Accuracy: {result['cv_accuracy_mean']:.3f}")
print(f"Top features: {result['feature_importance']}")
```

### `visualization.py` — Publication-Quality Figures

12 figure types, all generated at 150+ DPI with consistent styling.

| Function | Figure Type | Description |
|----------|-------------|-------------|
| `plot_msms_summary()` | Bar chart | MS/MS submitted vs identified per sample |
| `plot_protein_counts()` | Bar chart | Protein groups detected per group |
| `plot_missing_values()` | Heatmap | Missing value pattern across samples |
| `plot_intensity_distribution()` | Histogram | log2 intensity density per sample |
| `plot_replicate_correlation()` | Scatter | Replicate correlation with Pearson r |
| `plot_venn()` | Venn diagram | Protein overlap between 2-3 groups |
| `plot_volcano()` | Scatter | Volcano plot with significance coloring |
| `plot_allergen_heatmap()` | Clustered heatmap | Allergen proteins across samples |
| `plot_pca()` | Scatter | PCA colored by group |
| `plot_top_proteins()` | Horizontal bar | Top N most abundant proteins |

**Styling Standards:**
```python
plt.rcParams.update({
    'figure.dpi': 150,
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})
sns.set_style("whitegrid")
```

---

## CLI Reference

```
python maxquant_lcms_skill.py [OPTIONS]
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input` | Path to `proteinGroups.txt` | *required* (unless `--demo`) |
| `--input-type` | Input format: `maxquant` or `diann` | `maxquant` |
| `--metadata` | Sample metadata file (SDRF/CSV/TSV) | auto-detect |
| `--quant` | Quantification type: `iBAQ`, `lfq`, `intensity` | `iBAQ` |
| `--contrasts` | Group pairs: `"A,B;A,C"` | all pairwise |
| `--fc-threshold` | log2 fold-change cutoff | `1.0` |
| `--fdr` | FDR threshold | `0.05` |
| `--allergen-keywords` | Comma-separated search terms | built-in list |
| `--model` | Classifier: `svm`, `rf`, `none` | `none` |
| `--output` | Output directory | `./report` |
| `--demo` | Run with synthetic demo data | `false` |

---

## Supported Input Formats

### MaxQuant Output Files

| File | Required | Description |
|------|----------|-------------|
| `proteinGroups.txt` | **Yes** | Protein-level quantification (LFQ/iBAQ/Intensity) |
| `peptides.txt` | No | Peptide-level data |
| `evidence.txt` | No | PSM-level evidence with MBR info |
| `summary.txt` | No | Run-level QC statistics |
| `parameters.txt` | No | MaxQuant search parameters |

### Metadata File

A tab-separated or comma-separated file with at least:
- `sample_id` — matching raw file names in MaxQuant output
- `group` — experimental group assignment

Optional columns: `replicate`, `batch`, `condition`, `organism`

### Auto-Detection

If no metadata is provided, the skill auto-detects groups from iBAQ/LFQ column names by splitting on the last dash (e.g., `iBAQ Sample-1`, `iBAQ Sample-2` → group `Sample`).

---

## Output Structure

Every analysis run produces a self-contained report directory:

```
report/
├── analysis_report.md          # Comprehensive Markdown report
├── fig01_msms_summary.png      # MS/MS identification summary
├── fig02_proteins_per_group.png
├── fig03_missing_values.png
├── fig04_intensity_distribution.png
├── fig05_replicate_correlation.png
├── fig06_venn_diagram.png
├── fig07_volcano_GroupA_vs_GroupB.png
├── fig09_top20_proteins.png
├── fig10_allergen_heatmap.png
├── fig12_pca.png
├── tables/
│   ├── proteinGroups_filtered.csv
│   ├── allergen_proteins.csv
│   ├── diff_GroupA_vs_GroupB.csv
│   ├── taxonomy_summary.csv
│   └── model_results.csv
├── commands.sh                 # Exact command to reproduce this run
└── checksums.sha256            # SHA-256 of every input and output file
```

---

## Test Suite

The test suite contains **50 TDD-style tests** covering all modules:

```bash
python test_skill.py
```

| Category | Tests | What It Validates |
|----------|-------|-------------------|
| Filtering | 4 | Reverse/contaminant/site-only removal |
| FASTA Parsing | 5 | UniProt sp/tr headers, multi-entry, edge cases |
| Transformation | 3 | log2, zero handling, correctness |
| Imputation | 3 | NaN filling, downshift, preservation of observed |
| Allergen Codes | 3 | Direct FASTA, keyword fallback, unknown handling |
| Taxonomy | 4 | Shrimp, dust mite, bacteria, empty string |
| Quant Columns | 2 | Column name generation, existence check |
| Differential Abundance | 5 | Result shape, columns, p-value range |
| Significance | 4 | Up/Down/NS classification accuracy |
| BH Correction | 4 | Length, monotonicity, bounds |
| PCA | 3 | Output shape, variance ratio validity |
| Correlation | 3 | Dict return, r range, replicate consistency |
| End-to-End Demo | 7 | Report, checksums, figures, tables, content |

**Expected output:**
```
Test Results: 50 passed, 0 failed, 50 total
SUCCESS: All tests PASSED!
```

---

## Skill Specification (SKILL.md)

This project follows the [ClawBio](https://github.com/ClawBio/ClawBio) skill specification pattern. The `SKILL.md` file defines:

- **Inputs**: What files the skill accepts
- **Outputs**: What the skill produces (report, figures, tables, reproducibility bundle)
- **Methods**: Exact algorithms and their citations
- **Constraints**: What the skill does and does NOT do
- **Parameters**: All tunable parameters with defaults and ranges

The specification acts as a contract: the agent orchestrates the pipeline but does **not** improvise bioinformatics decisions beyond what is specified.

---

## Constraints & Guardrails

### This Skill DOES:
- Process MaxQuant and DIA-NN output files
- Perform differential abundance analysis with proper statistics
- Generate publication-quality visualizations
- Annotate proteins with allergen nomenclature and taxonomy
- Build predictive models for sample classification
- Produce reproducible analysis bundles

### This Skill DOES NOT:
- Process raw mass spectrometry data (.raw, .mzML)
- Perform peptide identification or database search
- Replace MaxQuant or DIA-NN search engines
- Provide clinical diagnostic recommendations
- Upload data to any cloud service

### Statistical Cautions:
- t-test reliability depends on sufficient replicates (>=3 per group recommended)
- Imputation assumes MNAR (missing not at random) — may not hold for all designs
- Multiple testing correction is essential for proteome-wide comparisons
- With only 2 replicates, p-values have very limited statistical power

### No Hallucinated Science:
- All methods based on established proteomics workflows
- Allergen codes follow WHO/IUIS nomenclature
- Statistical methods cite published references

---

## References

1. Cox J, Mann M. MaxQuant enables high peptide identification rates, individualized p.p.b.-range mass accuracies and proteome-wide protein quantification. *Nat Biotechnol*. 2008;26(12):1367-72.
2. Tyanova S, Temu T, Cox J. The MaxQuant computational platform for mass spectrometry-based shotgun proteomics. *Nat Protoc*. 2016;11(12):2301-19.
3. Giai Gianetto Q, et al. Uses and misuses of the fudge factor in quantitative discovery proteomics. *Proteomics*. 2016;16(14):1955-60.
4. Galaxy Training Network. Label-free data analysis using MaxQuant. GTN Tutorial GTN:T00218.
5. Keilhauer EC, Hein MY, Mann M. Accurate protein complex retrieval by affinity enrichment mass spectrometry (AE-MS). *Mol Cell Proteomics*. 2015;14(1):120-35.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
