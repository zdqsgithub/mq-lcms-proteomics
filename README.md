# MaxQuant LC-MS/MS Proteomics Bioinformatics & Modeling Skill

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-86%2F86%20passed-brightgreen.svg)](#test-suite)
[![v2.2](https://img.shields.io/badge/version-2.2-blue.svg)](#changelog)

A specification-constrained **agent skill** for end-to-end processing of MaxQuant LC-MS/MS proteomics data. Supports both **group comparison** and **time-course stability** analysis modes, with extensible allergen/taxonomy databases and vectorized statistics.

This skill integrates methodologies from five established open-source projects:

| Source | Contribution |
|--------|-------------|
| [Galaxy MaxQuant Tutorial](https://training.galaxyproject.org/training-material/topics/proteomics/tutorials/maxquant-label-free/tutorial.html) | Pipeline logic, filtering, QC |
| [ClawBio](https://github.com/ClawBio/ClawBio) | Skill specification, reproducibility bundles |
| [K-Dense-AI](https://github.com/K-Dense-AI/scientific-agent-skills) | Visualization standards |
| [Superpowers](https://github.com/obra/superpowers) | TDD, spec-first development |
| [Autoresearch](https://github.com/karpathy/autoresearch) | SVM/RF classifiers, optimization |

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Analysis Modes](#analysis-modes)
- [Architecture](#architecture)
- [Modules](#modules)
- [CLI Reference](#cli-reference)
- [External Databases](#external-databases)
- [Output Structure](#output-structure)
- [Test Suite](#test-suite)
- [Changelog (v1 → v2)](#changelog)
- [References](#references)
- [License](#license)

---

## Features

- **Three Analysis Modes**: Group comparison (`--mode comparison`), time-course stability (`--mode stability`), and deep characterization (`--mode deep-stability`)
- **Vectorized Statistics**: ~50-100x faster differential abundance via numpy broadcasting
- **Extensible Allergen DB**: JSON-based WHO/IUIS nomenclature covering crustacean, plant/pollen, mite, insect, pet, and food allergens
- **Extensible Taxonomy DB**: JSON-based species categorization for 13+ biological groups
- **Auto-Detection**: Automatically detects sample groups and quantification columns from MaxQuant output
- **20 Visualization Types**: Volcano, heatmap, PCA, Venn, time-course grids, waterfall charts, composition shifts, functional enrichment, MW distribution, oxidation heatmaps, degradation route summary
- **Reproducibility Bundle**: Every run generates `commands.sh` and `checksums.sha256`
- **Local-First**: All processing runs locally — no data uploaded anywhere

---

## Quick Start

```bash
git clone https://github.com/zdqsgithub/mq-lcms-proteomics.git
cd mq-lcms-proteomics

pip install -r requirements.txt

# Demo: group comparison
python maxquant_lcms_skill.py --demo --output demo_report

# Demo: stability mode
python maxquant_lcms_skill.py --demo --mode stability --output demo_stability

# Deep stability (includes oxidation, protease, pathway analysis)
python maxquant_lcms_skill.py --input txt/proteinGroups.txt --mode deep-stability --output report

# Run tests (76 tests)
python test_skill.py
```

---

## Installation

**Python 3.10+** required.

```bash
pip install -r requirements.txt
```

Dependencies: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, `matplotlib-venn`

---

## Analysis Modes

### Mode 1: Comparison (Default)

Standard group-vs-group differential abundance analysis.

```bash
python maxquant_lcms_skill.py \
  --input proteinGroups.txt \
  --quant iBAQ \
  --contrasts "Greer,Inhouse;Greer,Phadia" \
  --output report
```

**Produces:** Volcano plots, heatmaps, PCA, Venn diagrams, differential abundance tables.

### Mode 2: Stability (Time-Course)

Time-course degradation analysis with baseline normalization.

```bash
python maxquant_lcms_skill.py \
  --input proteinGroups.txt \
  --mode stability \
  --quant iBAQ \
  --output stability_report
```

**Produces:** Time-course profiles, waterfall charts, composition pie shifts, degradation rankings.

**Example:** W6 mugwort allergen thermal stability at 37°C — the skill auto-detects Day 0/3/7 groups, normalizes to baseline, classifies proteins as Degrading/Stable/Increasing, and identifies profilin/polcalcin degradation as the cause of potency loss.

### Mode 3: Deep Stability (v2.1+, updated v2.2)

Full stability analysis + functional enrichment + oxidation kinetics + deamidation sites + protease/degradation route characterization + coverage kinetics + sequence composition.

```bash
python maxquant_lcms_skill.py \
  --input txt/proteinGroups.txt \
  --mode deep-stability \
  --quant iBAQ \
  --output deep_report
```

**Produces:** Everything from stability mode PLUS functional enrichment bar charts, MW distributions, oxidation heatmaps, deamidation site analysis, protease inventory, semi-tryptic peptide analysis, coverage kinetics (unfolding evidence), sequence composition features (GRAVY, %Pro), and a 4-panel degradation route summary.

**v2.2 additions:**
- **Deamidation sites**: Parses `Deamidation (NQ)Sites.txt` if present, correlates with degradation
- **Coverage kinetics**: Tracks unique peptide count per protein per time point — distinguishes unfolding (coverage ↑) from aggregation (coverage ↓)
- **Sequence composition**: Identifies compositional features (e.g., %Proline) that predict which proteins degrade

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│           maxquant_lcms_skill.py (CLI)               │
│              Mode Dispatcher                         │
│  ┌────────────┐ ┌────────────┐ ┌────────────────┐    │
│  │ comparison │ │ stability  │ │ deep-stability │    │
│  └─────┬──────┘ └─────┬──────┘ └───────┬────────┘    │
├────────┴──────────────┴────────────────┴─────────────┤
│  core.py              stats_engine.py                │
│  ─ load/filter        ─ vectorized DE                │
│  ─ FASTA parsing      ─ timecourse_analysis()        │
│  ─ allergen_db.json   ─ BH-FDR, s0                   │
│  ─ taxonomy_db.json   ─ PCA, SVM/RF                  │
├─────────────────────────────────────────────────────┤
│  degradation_routes.py (v2.1+v2.2)                   │
│  ─ functional enrichment   ─ oxidation kinetics      │
│  ─ protease inventory      ─ semi-tryptic detection  │
│  ─ deamidation assessment  ─ peptide appearance      │
│  ─ coverage kinetics (v2.2) ─ deamidation sites(v2.2)│
│  ─ sequence composition(v2.2)─ peptide GRAVY  (v2.2) │
├─────────────────────────────────────────────────────┤
│  visualization.py                                    │
│  ─ 12 comparison plots  ─ 4 time-course plots        │
│  ─ 4 degradation route plots (v2.1 NEW)              │
├─────────────────────────────────────────────────────┤
│  Reproducibility: commands.sh + checksums.sha256     │
└─────────────────────────────────────────────────────┘
```

---

## Modules

### `core.py` — Data Engine

| Function | Description |
|----------|-------------|
| `load_maxquant(data_dir)` | Load all MaxQuant output files |
| `filter_protein_groups(df)` | Remove reverse/contaminant/site-only |
| `extract_description(header)` | Parse UniProt FASTA headers |
| `auto_detect_groups(pg, quant)` | Auto-detect groups from column names |
| `get_quant_columns(df, groups)` | Get iBAQ/LFQ/intensity columns |
| `log2_transform(df, cols)` | Log2 with zero→NaN |
| `impute_missing(df)` | Down-shifted Gaussian (MNAR) |
| `get_allergen_code(header, desc)` | WHO/IUIS mapping via `allergen_db.json` |
| `categorize_taxonomy(name)` | Species grouping via `taxonomy_db.json` |

### `stats_engine.py` — Statistical Analysis

| Function | Description |
|----------|-------------|
| `differential_abundance()` | **Vectorized** Welch's t-test (v2: ~50-100x faster) |
| `timecourse_analysis()` | **NEW** — baseline normalization, trend classification |
| `benjamini_hochberg()` | FDR correction |
| `classify_significance()` | Up/Down/NS classification |
| `run_pca()` | PCA dimensionality reduction |
| `train_classifier()` | SVM/RF with LOO-CV |

### `degradation_routes.py` — Degradation Characterization (v2.1+v2.2)

| Function | Description | Version |
|----------|-------------|:-------:|
| `assign_functional_category()` | Classify proteins into 14 functional categories | v2.1 |
| `functional_enrichment()` | Enrichment analysis across Degrading/Stable/Increasing | v2.1 |
| `analyze_oxidation_sites()` | Parse Oxidation (M)Sites.txt, compute kinetics | v2.1 |
| `correlate_oxidation_degradation()` | Pearson correlation: modification vs stability | v2.1 |
| `detect_semi_tryptic()` | Classify peptides as fully/semi/non-tryptic | v2.1 |
| `semi_tryptic_kinetics()` | Track protease activity over time | v2.1 |
| `inventory_proteases_phosphatases()` | Catalog endogenous proteases with risk level | v2.1 |
| `peptide_appearance()` | Detect new/lost peptides (clipping products) | v2.1 |
| `count_deamidation_motifs()` | Count NG/NS/NT deamidation hotspots | v2.1 |
| `peptide_gravy()` | Compute Kyte-Doolittle hydropathy score | **v2.2** |
| `coverage_kinetics()` | Track unique peptides per protein per TP (unfolding evidence) | **v2.2** |
| `analyze_deamidation_sites()` | Parse Deamidation (NQ)Sites.txt, compute kinetics | **v2.2** |
| `sequence_composition()` | Per-protein GRAVY, aliphatic index, %Pro, %charged | **v2.2** |

### `visualization.py` — 20 Figure Types

**Comparison mode:** MS/MS summary, protein counts, missing values, intensity distribution, replicate correlation, Venn diagram, volcano, allergen heatmap, PCA, top proteins

**Stability mode:** Time-course grid, waterfall chart, composition shift, grouped bar

**Deep stability (v2.1 NEW):** Functional enrichment bar, MW by trend, oxidation heatmap, 4-panel degradation routes

---

## CLI Reference

```
python maxquant_lcms_skill.py [OPTIONS]
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input` | Path to `proteinGroups.txt` | required (unless `--demo`) |
| `--input-dir` | MaxQuant `txt/` directory | auto from `--input` |
| `--mode` | `comparison`, `stability`, or `deep-stability` | `comparison` |
| `--quant` | `iBAQ`, `lfq`, or `intensity` | `iBAQ` |
| `--contrasts` | Group pairs: `"A,B;A,C"` | all pairwise |
| `--fc-threshold` | log2 fold-change cutoff | `1.0` |
| `--fdr` | FDR threshold | `0.05` |
| `--model` | `svm`, `rf`, or `none` | `none` |
| `--output` | Output directory | `./report` |
| `--demo` | Run with synthetic data | `false` |

---

## External Databases

### `allergen_db.json`

Extensible allergen nomenclature database. Add new allergen families by editing the JSON:

```json
{
  "organism_codes": { "ARTVU": "Art v", "BETPN": "Bet v", ... },
  "keyword_groups": {
    "profilin": { "group": "4", "category": "pan-allergen" },
    "polcalcin": { "group": "5", "category": "calcium-binding" },
    ...
  }
}
```

**Coverage:** Crustacean (Pen a/v/m, Mac r, Cra c), Plant/Pollen (Art v, Amb a, Bet v, Ole e, Phl p), Mite (Der p/f), Pet (Fel d, Can f), Insect (Api m, Ves v), Food (Ara h, Tri a).

### `taxonomy_db.json`

Species categorization rules:

```json
{
  "categories": {
    "Mugwort/Artemisia": ["Artemisia"],
    "Birch": ["Betula", "Alnus", "Corylus"],
    "Grass Pollen": ["Lolium", "Phleum", "Dactylis"],
    ...
  }
}
```

---

## Output Structure

### Comparison Mode
```
report/
├── analysis_report.md
├── fig02_proteins_per_group.png
├── fig05_replicate_correlation.png
├── fig07_volcano_*.png
├── fig10_allergen_heatmap.png
├── fig12_pca.png
├── tables/
│   ├── proteinGroups_filtered.csv
│   ├── diff_GroupA_vs_GroupB.csv
│   └── allergen_proteins.csv
├── commands.sh
└── checksums.sha256
```

### Stability Mode
```
report/
├── stability_report.md
├── fig_timecourse_profiles.png
├── fig_waterfall.png
├── fig_grouped_bar.png
├── fig_composition.png
├── fig10_allergen_heatmap.png
├── tables/
│   ├── stability_summary.csv
│   └── proteinGroups_filtered.csv
├── commands.sh
└── checksums.sha256
```

### Deep Stability Mode (v2.1)
```
report/
├── stability_report.md            # Includes deep analysis appendix
├── fig_functional_enrichment.png  # Functional category enrichment
├── fig_mw_by_trend.png            # MW distribution by trend
├── fig_oxidation_heatmap.png      # Top oxidation sites
├── fig_degradation_routes.png     # 4-panel summary
├── tables/
│   ├── stability_summary.csv
│   ├── oxidation_sites.csv
│   └── proteinGroups_filtered.csv
├── commands.sh
└── checksums.sha256
```

---

## Test Suite

**76 tests** covering all modules:

```bash
python test_skill.py
```

| Category | Tests | v2 New? |
|----------|-------|---------|
| Filtering | 4 | |
| FASTA Parsing | 5 | |
| Log2 Transform | 2 | |
| Imputation | 3 | |
| Allergen Codes (crustacean) | 1 | |
| **Allergen Codes (plant/pollen)** | **5** | **Yes** |
| Taxonomy (shrimp/mite/bacteria) | 3 | |
| **Taxonomy (mugwort/ragweed/birch/grass)** | **5** | **Yes** |
| **Auto-detect Groups** | **2** | **Yes** |
| Quant Columns | 2 | |
| Vectorized DE | 5 | Rewritten |
| **Timecourse Analysis** | **6** | **Yes** |
| Significance | 3 | |
| BH Correction | 3 | |
| PCA | 2 | |
| Correlation | 2 | |
| **End-to-end Comparison** | **3** | **Yes** |
| **End-to-end Stability** | **3** | **Yes** |
| **Functional Categories** | **5** | **v2.1** |
| **Functional Enrichment** | **3** | **v2.1** |
| **Semi-tryptic Detection** | **3** | **v2.1** |
| **Protease Inventory** | **3** | **v2.1** |
| **Deamidation Motifs** | **1** | **v2.1** |
| **Peptide Appearance** | **2** | **v2.1** |
| **GRAVY Score** | **3** | **v2.2** |
| **Coverage Kinetics** | **3** | **v2.2** |
| **Sequence Composition** | **4** | **v2.2** |

---

## Changelog

### v2.2 (Current)

- **Coverage kinetics** — Track unique peptide count per protein per time point; distinguishes thermal unfolding (coverage ↑ despite abundance ↓) from aggregation/precipitation (coverage ↓)
- **Deamidation site analysis** — Parse `Deamidation (NQ)Sites.txt`, compute per-site kinetics, correlate with protein degradation
- **Sequence composition** — Per-protein GRAVY, aliphatic index, %Proline, %charged, %hydrophobic; statistical comparison across stability trends (Mann-Whitney U)
- **Peptide GRAVY scoring** — Kyte-Doolittle hydropathy for individual peptides and protein-level aggregation
- **Deep-stability pipeline** now runs 7 analysis steps (was 4): stability → enrichment → MW → oxidation → deamidation → protease/coverage → composition
- **86 tests** (up from 76)

### v2.1

- **`--mode deep-stability`** — Full stability + pathway + oxidation + protease analysis in one command
- **`degradation_routes.py`** — New module with 9 functions for degradation characterization
- **Functional enrichment analysis** — 14 categories, enrichment ratios across stability trends
- **Oxidation kinetics** — Parse `Oxidation (M)Sites.txt`, correlation with degradation
- **Protease inventory** — Detect endogenous proteases with risk classification (HIGH/MODERATE/LOW)
- **Semi-tryptic peptide analysis** — Evidence-based protease activity detection
- **Deamidation motif scanning** — Count NG/NS/NT hotspots
- **4 new visualization functions** — Functional enrichment, MW, oxidation heatmap, 4-panel degradation
- **76 tests** (up from 59)

### v2.0

- **Vectorized `differential_abundance()`** — numpy broadcasting replaces `iterrows()` loop (~50-100x speedup)
- **`--mode stability`** — New time-course degradation analysis mode with baseline normalization
- **External `allergen_db.json`** — Extensible allergen mapping covering 30+ protein families
- **External `taxonomy_db.json`** — 13 biological groups including plants, pollen, fungi
- **4 new visualization functions** — Time-course grid, waterfall, composition shift, grouped bar
- **`auto_detect_groups()`** — No metadata needed for standard MaxQuant naming conventions
- **`timecourse_analysis()`** — Vectorized trend computation with p-values
- **59 tests** (up from 50)

### v1.0

- Initial release with comparison mode, 12 visualizations, 50 tests

---

## References

1. Cox J, Mann M. MaxQuant enables high peptide identification rates. *Nat Biotechnol*. 2008;26(12):1367-72.
2. Tyanova S, Temu T, Cox J. The MaxQuant computational platform. *Nat Protoc*. 2016;11(12):2301-19.
3. Giai Gianetto Q, et al. Uses and misuses of the fudge factor. *Proteomics*. 2016;16(14):1955-60.
4. Galaxy Training Network. Label-free data analysis using MaxQuant. GTN:T00218.
5. Keilhauer EC, Hein MY, Mann M. Accurate protein complex retrieval by AE-MS. *MCP*. 2015;14(1):120-35.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
