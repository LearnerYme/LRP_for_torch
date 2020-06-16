# LRP for Pytorch
Layer-wise Relevance Propagate For Pytorch.

Author: Yme

***

A demo is uploaded. It is quite easy to understand the usage of this LRP tool box, I think. However, I'd like to show a brief tutorial as well.

**USAGE**

Step 1, import lrp tool box and your network, then load your '.pth' file.

Step 2, create an instance of lrp, like: `lrpt = lrp(1e-6)`. Here only ask for 1 argument (epsilon).

Step 3, use `lrpt.Rinit(network, input, False)` to initializing R. If your network contains softmax in your last layer, set False here, and if not, set True.

Step 4, start from your last layer, set parameters for lrpt, like: `lrpt.set_('fc',lenet.fc2_value,lenet.fc3.weight)`.

Step 5, then process lrp by, for example, `lrpt.fc()`.

Step 6, repeat step 4 and 5 until you arrive at input layer.

***

Some formulas are not so clear for me, for example, should we plus bias term in the denominator? I think it should not be a 'b' there if we want to keep ![](https://latex.codecogs.com/svg.latex?\sum_i{R_i}^{(l)}=\sum_j{R_j}^{(l+1)}). And so does the author of this paper.

Homanga Bharadhwaj. Layer-wise Relevance Propagation for Explainable Recommendations. (arXiv:1807.06160v1 [cs.LG] 17 Jul 2018).

However, the origin paper give a formula with bias.

Sebastian Bach, et. al.. On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation. (DOI:10.1371/journal.pone.0130140).

and

Sebastian Lapuschkin, et. al.. The LRP Toolbox for Artificial Neural Networks. (Journal of Machine Learning Research 17 (2016) 1-5).

My codes are based on this project: https://github.com/sebastian-lapuschkin/lrp_toolbox with some modifying.

This is my first project, if you find any problems, please tell me and thanks a lot.

Hope this tool box can help you. ^^