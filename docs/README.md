# How to Benchmark Vision Foundation Models for Semantic Segmentation?

**Authors:** [Tommie Kerssies](https://tommiekerssies.com/), [Daan de Geus](https://ddegeus.github.io/), [Gijs Dubbelman](https://www.linkedin.com/in/gijsdubbelman/)  
**Affiliation:** Eindhoven University of Technology  
**Contact:** {t.kerssies, d.c.d.geus, g.dubbelman}@tue.nl  
**Publication:** CVPR 2024 Workshop Proceedings for the Second Workshop on Foundation Models  
**Paper:** [arXiv](https://arxiv.org/abs/2404.12172)  
**Code**: [GitHub](https://github.com/tue-mps/benchmark-vfm-ss)

## Abstract
Recent vision foundation models (VFMs) have demonstrated proficiency in various tasks but require supervised fine-tuning to perform the task of semantic segmentation effectively. Benchmarking their performance is essential for selecting current models and guiding future model developments for this task. The lack of a standardized benchmark complicates comparisons. Therefore, the primary objective of this paper is to study how VFMs should be benchmarked for semantic segmentation. To do so, various VFMs are fine-tuned under various settings, and the impact of individual settings on the performance ranking and training time is assessed. Based on the results, the recommendation is to fine-tune the ViT-B variants of VFMs with a 16x16 patch size and a linear decoder, as these settings are representative of using a larger model, more advanced decoder and smaller patch size, while reducing training time by more than 13 times. Using multiple datasets for training and evaluation is also recommended, as the performance ranking across datasets and domain shifts varies. Linear probing, a common practice for some VFMs, is not recommended, as it is not representative of end-to-end fine-tuning. The benchmarking setup recommended in this paper enables a performance analysis of VFMs for semantic segmentation. The findings of such an analysis reveal that pretraining with promptable segmentation is not beneficial, whereas masked image modeling (MIM) with abstract representations is crucial, even more important than the type of supervision used.

## Citation
```
@inproceedings{kerssies2024benchmarking,
  author={Kerssies, Tommie and de Geus, Daan and Dubbelman, Gijs},
  title={How to Benchmark Vision Foundation Models for Semantic Segmentation?},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  year={2024},
}
```
