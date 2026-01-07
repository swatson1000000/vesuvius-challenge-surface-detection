# Vesuvius Challenge - Surface Detection: Official Competition Information

**Downloaded from**: https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection  
**Date**: January 5, 2026  
**Status**: Active (1 month remaining)

---

## Competition Overview

### Description

The library at the Villa dei Papiri is one-of-a-kind: it's the only classical antiquity known to survive. But when Mount Vesuvius erupted in AD 79, most of its scrolls were turned into carbonized bundles of ash. Nearly 2,000 years later, many are still sealed shut, too delicate to unroll and too complex to decode. In this competition, you'll build a model to segment the scroll's surface in CT scans, a critical step in revealing the texts hidden inside.

Physical unrolling would destroy the scrolls, and today's digital methods can handle only the easy parts, like clean, well-spaced layers. But the tightest, most tangled areas are often where the real discoveries hide. To recover those, we need better segmentation to handle noise and compression without distorting the scroll's shape.

You'll work with real 3D CT data from the previous Vesuvius Challenge, which uncovered passages from sealed scrolls, including an ancient book title revealed for the first time. Your model's job is to trace the scroll's surface as it winds through folds, gaps, and distortions so the next stage of virtual unwrapping can do its magic.

Your efforts could help unlock works of philosophy, poetry, and history that haven't been read in centuries. Texts that might have been lost forever—until now.

Let's bring them back, one layer at a time.

### Previous Competition
- Previous Vesuvius Challenge: https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection

---

## Timeline

- **November 13, 2025** - Start Date
- **February 6, 2026** - Entry Deadline (must accept rules before this date)
- **February 6, 2026** - Team Merger Deadline (last day to join or merge teams)
- **February 13, 2026** - Final Submission Deadline

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

---

## Prizes

- **1st place**: $60,000
- **2nd place**: $40,000
- **3rd place**: $30,000
- **4th place**: $20,000
- **5th place**: $15,000
- **6th place**: $10,000
- **7th place**: $10,000
- **8th place**: $5,000
- **9th place**: $5,000
- **10th place**: $5,000

**Total Prize Pool**: $200,000

---

## Dataset Description

The dataset includes 3D chunks of binary labeled CT scans of the closed and carbonized Herculaneum scrolls. Data was acquired at the ESRF synchrotron in Grenoble, on beamline BM18 and at the DLS synchrotron in Oxford, on beamline I12.

The dimension of a sample chunk is not fixed, and can vary across the dataset.

### Papyrus Structure

A papyrus sheet is composed of two layers:
- **Recto**: Surface facing the umbilicus (center of scroll) - composed of horizontal fibers
- **Verso**: Outer surface - composed of vertical fibers

### Labeling Goals

The ideal solution is detecting the **recto surface** – the surface of the papyrus sheet that faces the umbilicus (the center of the scroll). The recto surface lies on the layer composed of horizontal fibers.

Since the scrolls survived a carbonized eruption, the sheets can be partially damaged and frayed. Detecting, within a reasonable approximation, just the position of a sheet, no matter whether the segmentation surface encompasses both the recto and the verso, is also fine for the purpose of virtually unwrapping the scrolls.

### Critical Requirements

**Avoid topological mistakes:**
- Artificial mergers between different sheets
- Holes that split a single entity in several disconnected components

The output of the machine learning model can be directly used in the Vesuvius Challenge pipeline to digitally unwrap (and read) the Herculaneum scrolls.

### Data Updates

Because the host team is laser-focused on unwrapping these scrolls, their research will continue throughout the Kaggle competition. Their annotation team will semi-manually unwrap additional portions of papyrus, and they expect to release both more labeled data AND ancient text as the competition progresses.

**Important**: Unlike the original training set, which underwent careful proofreading and refinement, these new labels will be less curated. Please use your judgment to decide how (and whether) each additional chunk should be incorporated into model training.

### Tutorial

Latest tutorial on digital unwrapping pipeline: https://www.youtube.com/watch?v=yHbpVcGD06U

---

## Evaluation

This competition uses a **weighted average of three different segmentation metrics**: Surface Dice, TopoScore, and VOI.

### Metric Formula

```
Final Score = 0.30 × TopoScore + 0.35 × SurfaceDice@τ + 0.35 × VOI_score
```

All metrics are bounded in [0, 1] with higher being better.

### 1. SurfaceDice@τ — Surface Proximity

A surface-aware variant of Dice: it scores the fraction of surface points (both prediction and ground truth) that lie within a spatial tolerance τ of each other. This respects that both labels and predictions may be off by a voxel or two while still capturing the right geometry.

**Computation essentials:**
- Extract voxelized surfaces of prediction (P) and ground truth (G)
- Compute nearest-surface distances with case spacing
- Count matches when distance ≤ τ
- Average matches both ways (P↔G)

**Edge cases:**
- Both empty → 1.0
- Exactly one empty → 0.0

**Defaults:**
- τ = 2.0 (in spacing units)

### 2. VOI_score — Instance Split/Merge Correctness

Variation of Information (VOI) compares the connected-component labelings of prediction and ground truth.

**Components:**
- VOI_split = H(GT | Pred) - tracks over-segmentation (splits)
- VOI_merge = H(Pred | GT) - tracks under-segmentation (mergers)
- VOI_total = VOI_split + VOI_merge

**Score conversion:**
```
VOI_score = 1 / (1 + α × VOI_total)
```
where α = 0.3

**Computation essentials:**
- Run 3D connected components (default 26-connectivity) on union foreground
- Compute VOI on these integer labelings
- Map to VOI_score

**Interpretation:**
- Responds strongly to component splits/merges
- Complements the shape-level guarantees of TopoScore

### 3. TopoScore — Topological Feature Preservation

Using Betti number matching from algebraic topology, we compare the topological features present in prediction vs. ground truth in each homology dimension k:
- **k=0**: Components
- **k=1**: Tunnels/handles
- **k=2**: Cavities

We compute a per-dimension Topological-F1 based on matched features, then take a weighted average over active dimensions.

**Default weights:**
- w0 = 0.34 (components)
- w1 = 0.33 (handles)
- w2 = 0.33 (cavities)

Inactive dimensions (no features in both) are skipped with renormalization.

**Why this helps:**
- Detects bridges across neighboring wraps (extra k=1)
- Detects within-wrap breaks (extra k=0)
- Detects spurious cavities (extra k=2)

### Metric Comparison Table

| Error Type | SurfaceDice@τ | VOI_score | TopoScore |
|------------|---------------|-----------|-----------|
| Slight boundary misplacement (≤ τ) | Tolerant (unchanged) | Usually unaffected | Unaffected |
| Artificial bridge between parallel layers | Local effect (may stay high) | ↓ Merge penalty (under-segmentation) | ↓ k=0/k=1 penalized |
| Split of same wrap into pieces | Local effect (may stay high) | ↓ Split penalty (over-segmentation) | ↓ k=0 penalized |
| Spurious holes/handles in sheet | Local effect (may stay high) | Small or none | ↓ k=1/k=2 penalized |

### Evaluation Details & Edge Cases

- **Binarization**: Foreground is non-zero by default (or x > threshold if specified)
- **Ignore regions**: Voxels marked ignored (via GT label or explicit mask) are set to background in both prediction and GT before all computations
- **Connectivity**: Default 26 in 3D (VOI only)
- **Spacing & tolerance**: All distances use per-case spacing (sz, sy, sx); τ is in the same physical units
- **Empty-mask conventions**:
  - Both empty → SurfaceDice=1, VOI_score=1, TopoScore=1
  - One empty → SurfaceDice=0, VOI_score near 0, TopoScore=0 on k=0

### Practical Tips for Competitors

- **Favor continuity within a wrap**; avoid bridges across wraps (thin spurious connections are costly in VOI and TopoScore)
- **Mind connectivity**: Thinning/dilation can alter counts; check results under 26-connectivity
- Use the demo notebook to validate: https://www.kaggle.com/code/sohier/vesuvius-2025-metric-demo/
- Metric resources: https://www.kaggle.com/datasets/sohier/vesuvius-metric-resources

**Note**: This metric takes a relatively long time to run; you may need to wait several hours for scoring to complete.

---

## Submission File

You must submit a ZIP containing one .tif volume mask per test image.

**Requirements:**
- Each mask must be named `[image_id].tif`
- Must match the dimensions of the source image exactly
- Use the same data type as the train mask (uint8)

---

## Code Requirements

Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met:

### Technical Requirements
- **CPU Notebook** ≤ 9 hours run-time
- **GPU Notebook** ≤ 9 hours run-time
- **Internet access** disabled
- **External data** freely & publicly available is allowed, including pre-trained models
- **Submission file** must be named `submission.zip`

### Resources
- Code Competition FAQ: https://www.kaggle.com/docs/competitions#notebooks-only-FAQ
- Code debugging doc: https://www.kaggle.com/code-competition-debugging

---

## References

### Key Papers

1. **Pitfalls of topology-aware image segmentation** - Berger et al., IPMI 2025 (2025)
   - https://arxiv.org/pdf/2412.14619v1

2. **Deep learning to achieve clinically applicable segmentation of head and neck anatomy for radiotherapy** - Nikolov et al., J Med Internet Res. 2021 Jul 12;23(7):e26151 (2021)
   - https://arxiv.org/pdf/1809.04430

3. **Efficient Betti Matching Enables Topology-Aware 3D Segmentation via Persistent Homology** - Stucki et al., arXiv preprint (2024)
   - https://arxiv.org/pdf/2407.04683

4. **clDice - a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation** - Shit et al., CVPR 2021 (2021)
   - https://openaccess.thecvf.com/content/CVPR2021/papers/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.pdf

5. **Efficient Connectivity-Preserving Instance Segmentation with Supervoxel-Based Loss Function** - Grim et al., AAAI-25 (2025)
   - https://ojs.aaai.org/index.php/AAAI/article/view/32326/34481

6. **Skeleton Recall Loss for Connectivity Conserving and Resource Efficient Segmentation of Thin Tubular Structures** - Kirchhoff et al., ECCV 2024 (2024)
   - https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09904.pdf

---

## Competition Statistics

- **Host**: Vesuvius Challenge
- **Prize Pool**: $200,000
- **Participation**:
  - 5,631 Entrants
  - 631 Participants
  - 560 Teams
  - 6,187 Submissions (as of data capture)

### Tags
- History
- Image Segmentation
- Segmentation
- Custom Metric

---

## Citation

```
Sean Johnson, David Josey, Elian Rafael Dal Prà, Hendrik Schilling, Youssef Nader,
Johannes Rudolph, Forrest McDonald, Paul Henderson, Giorgio Angelotti, Sohier Dane,
and María Cruz. Vesuvius Challenge - Surface Detection.
https://kaggle.com/competitions/vesuvius-challenge-surface-detection, 2025. Kaggle.
```

---

## Important Links

- **Competition Home**: https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection
- **Metric Demo Notebook**: https://www.kaggle.com/code/sohier/vesuvius-2025-metric-demo/
- **Metric Resources Dataset**: https://www.kaggle.com/datasets/sohier/vesuvius-metric-resources
- **Previous Competition**: https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection
- **Digital Unwrapping Tutorial**: https://www.youtube.com/watch?v=yHbpVcGD06U
- **Vesuvius Challenge Profile**: https://www.kaggle.com/organizations/scrollprize

---

## Summary of Key Requirements

### Model Objectives
1. Segment papyrus surfaces in 3D CT scans
2. Maintain topological integrity (no mergers between layers, no splits within layers)
3. Handle noise, compression, and distortions
4. Optimize for three metrics: TopoScore (30%), SurfaceDice@2.0 (35%), VOI_score (35%)

### Constraints
- 9-hour inference time limit
- No internet access during inference
- Must work with variable-sized 3D volumes
- Output must match input dimensions exactly

### Success Factors
- Topology-aware model design and post-processing
- Robust to scroll variation and scan quality
- Efficient inference pipeline
- Balance between three distinct metrics

---

**Document Last Updated**: January 5, 2026
