---
layout: default
title: "How to Benchmark Vision Foundation Models for Semantic Segmentation?"
---

# How to Benchmark Vision Foundation Models for Semantic Segmentation?

**Published at:** Second Workshop on Foundation Models at CVPR 2024

## Abstract
Recent vision foundation models (VFMs) have demonstrated proficiency in various
tasks but require fine-tuning with semantic mask labels for the task of
semantic segmentation. Benchmarking their performance is essential for
selecting current models and guiding future model developments for this task.
The lack of a standardized benchmark complicates comparisons, therefore the
primary objective of this paper is to study how VFMs should be benchmarked for
this task. To do so, various VFMs are fine-tuned under several settings, and
the impact of individual settings on the performance ranking and training time
is assessed. Based on the results, the recommendation is to fine-tune the ViT-B
variants of VFMs with a 16$\times$16 patch size and a linear decoder, as these
settings are representative of using a larger model, more advanced decoder and
smaller patch size, while reducing training time by more than 13$\times$. Using
multiple datasets for training and evaluation is also recommended, as the
performance ranking across datasets and domain shifts varies. Linear probing, a
common practice for some VFMs, is not recommended, as it is not representative
of end-to-end fine-tuning. The recommended benchmarking setup enables a
performance analysis of VFMs for semantic segmentation. The findings of such an
analysis reveal that promptable segmentation pretraining is not beneficial,
whereas masked image modeling (MIM) pretraining with abstract representations
appears crucial, even more so than the type of supervision.
