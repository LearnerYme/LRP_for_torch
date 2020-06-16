# LRP for Pytorch
Layer-wise Relevance Propagate For Pytorch.

Author: Yme

Currently, this is a test version. I'll upload usage and demo later.

Some formulas are not so clear for me, for example, should we plus bias term in the denominator? I think it should not be a 'b' there if we want to keep $\sum_i^{(l)}{R_i}=\sum_j^{(l+1)}{R_j}$. And so does the author of this paper.

Homanga Bharadhwaj. Layer-wise Relevance Propagation for Explainable Recommendations. (arXiv:1807.06160v1 [cs.LG] 17 Jul 2018).

However, the origin paper give a formula with bias.

Sebastian Bach, et. al.. On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation. (DOI:10.1371/journal.pone.0130140).

and

Sebastian Lapuschkin, et. al.. The LRP Toolbox for Artificial Neural Networks. (Journal of Machine Learning Research 17 (2016) 1-5).

My codes are based on this project: https://github.com/sebastian-lapuschkin/lrp_toolbox.

This is my first project, if you find any problems, please tell me and thanks a lot.