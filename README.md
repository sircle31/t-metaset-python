# t-METASET: Python Implementation

## Introduction
This repository is intended to publicly share a Python implementation of the t-METASET framework, 
proposed in the paper entitled [t-METASET: Task-Aware Acquisition of Metamaterial Datasets through Diversity-based Active Learning](https://asmedigitalcollection.asme.org/mechanicaldesign/article/145/3/031704/1147447/t-METASET-Task-Aware-Acquisition-of-Metamaterial). 
**t-METASET** is an abbreviation of _task-aware_ METASET, inspired by [METASET](https://asmedigitalcollection.asme.org/mechanicaldesign/article/143/3/031707/1087582/METASET-Exploring-Shape-and-Property-Spaces-for) proposed by Chan et al.
For general inquiry contact the first author Doksoo Lee (doksoo.lee@northwestern.edu).

## License
This code is licensed under the MIT license. It is encouraged to use all or portions for your research or related projects, provided that you include the following citation information, or any alternative forms that give a proper credit to:

```bibtex
@article{lee2023t,
  title={T-METASET: Task-aware acquisition of metamaterial datasets through diversity-based active learning},
  author={Lee, Doksoo and Chan, Yu-Chin and Chen, Wei and Wang, Liwei and van Beek, Anton and Chen, Wei},
  journal={Journal of Mechanical Design},
  volume={145},
  number={3},
  pages={031704},
  year={2023},
  publisher={American Society of Mechanical Engineers}
}
```


## Component I. Shape Descriptor (Variational Autoencoder)

Given a massive set of high-dimensional metamaterial unit cells, we distill a compact yet expressive shape descriptor of unit cells through variational autoencoder ([Kingma et al., arXiv, 2013](https://arxiv.org/abs/1312.6114)), which entails data-driven feature extraction through unsupervised representation learning, following [Wang et al., CMAME, 2020](https://www.sciencedirect.com/science/article/pii/S0045782520305624).

For the `t-METASET` class included in `main.py`, it can take other shape descriptors as well. The choice is subject to the regressor that serves as an active learner in the `t-METASET` procedure. For user-defined low-dimensional descriptors (say less than 10-D), the GP regressor set as default is expected to serve well upon proper feature-wise scaling.

## Component II. Regressor (Gaussian Processes)

We disclose that `GP_pytorch.py` includes snippets for the implementation of [Batch Independent Multioutput Gaussian Processes](https://docs.gpytorch.ai/en/stable/examples/03_Multitask_Exact_GPs/Batch_Independent_Multioutput_GP.html) availabe in [gpytorch](https://gpytorch.ai/),
whose key reference is:

- Gardner, J., Pleiss, G., Weinberger, K. Q., Bindel, D., & Wilson, A. G. (2018). [Gpytorch: Blackbox matrix-matrix gaussian process inference with gpu acceleration](https://papers.nips.cc/paper/2018/hash/101c673e6e65ea618ce8c964f23f10d6-Abstract.html). Advances in neural information processing systems, 31.

In this version, we do not consider any accelerations involved with access to GPU, and room for scalability improvement with respect to cardinality through sparse variational approximations (e.g., Titsias, 2009). Nor do we consider any a-priori filtering of training data based on distances in the input space, which could improve stability of the GP training. This simple implementation of gpytorch is justified for our purposes since leveraging the gpytorch implementation in python is inessential to demonstrate the core idea of the `t-METASET` – creating an active learning loop interlocked with descriptor, learner, and sampler, and then maneuvering the sampling procedure to control the property distribution at will.

Note that the choice of regressor for the `t-METASET` procedure is not limited by GPs, but open to other surrogate models. It is generally advised to choose the regressor with the shape descriptor taken into account. Since the proposed sampling criteria do not demand uncertainty, a variety of models is allowed to be considered, e.g., standard neural nets.

## Component III. Diversity Sampler (Determinantal Point Processes)

We disclose that `DPP.py` builds on a python translation of Determinantal Point Processes from Alex Kuelsza’s matlab implementation, associated with the following reference:

- Kulesza, A., & Taskar, B. (2012). [Determinantal point processes for machine learning](https://arxiv.org/abs/1207.6083). Foundations and Trends® in Machine Learning, 5(2–3), 123-286.

The code development is built on the following key references as well:

- Kulesza, A., & Taskar, B. (2011). [k-DPPs: Fixed-size determinantal point processes](https://dl.acm.org/doi/10.5555/3104482.3104632). In Proceedings of the 28th International Conference on Machine Learning (ICML-11) (pp. 1193-1200).
- Affandi, R. H., Kulesza, A., & Fox, E. B. (2012). [Markov determinantal point processes](https://arxiv.org/abs/1210.4850). arXiv preprint arXiv:1210.4850.
 - Gillenwater, J. A., Kulesza, A., Fox, E., & Taskar, B. (2014). [Expectation-maximization for learning determinantal point processes](https://papers.nips.cc/paper/2014/hash/3a9ecb25d24f9d0a8279b19d014d9c68-Abstract.html). Advances in Neural Information Processing Systems, 27.
- Gartrell, M., Brunel, V. E., Dohmatob, E., & Krichene, S. (2019). [Learning nonsymmetric determinantal point processes](https://papers.nips.cc/paper/2019/hash/9a382a7b26e99e2c12af3abf8b3b7f0d-Abstract.html). Advances in Neural Information Processing Systems, 32.

The exhaustive list of references is available in our paper.

## The t-METASET Procedure: Python Implementation

The numerical experiment presented in our paper was based on matlab implementation, primarily built on [Kulesza and Taskar, arXiv, 2011](https://arxiv.org/abs/1207.6083) for DPP and [Bostanabad et al. JMD, 2019](https://asmedigitalcollection.asme.org/mechanicaldesign/article/141/11/111402/955350/Globally-Approximate-Gaussian-Processes-for-Big) for GP. On the other hand, the python translation and its numerical results provided here are subject to discrepancies due to several factors, e.g., eigendecomposition by `numpy.linalg.eigh`, GP implementation through `gpytorch`.

Unlike the matlab implementation in the paper, only the first roughness convergence is considered for the example numerical results, conducted for 2D orthotropic mechanical metamaterial dataset ([Wang et al., SMO, 2019](https://link.springer.com/article/10.1007/s00158-020-02523-5)). In our numerical experiments, we use the precomputed data, whose the effective properties have been fully evaluated, so that we avoid the design evaluation ([Xia and Breitkopf, SMO, 2015](https://link.springer.com/article/10.1007/s00158-015-1294-0)) that could be time-consuming and focus on the proposed task-aware data acquisition. It is important to note that the t-METASET algorithm per se has no access to the property data at the beginning, but will incrementally collect samples batch by batch and estimate the properties of unseen unit cells through the learner, which a GP unless replaced otherwise.

## Access to the Test Datasets used in the Paper

The full data used for the numerical experiment in the paper is available [here](https://ideal.mech.northwestern.edu/software/) (under 'Datasets') upon official request. In this repo, part of the 2D orthotropic mechanical metamaterial dataset is provided under `./data` for quick reproduction of the result of Task II-2 — (promoting a dataset biased towards high elastic anisotropy — in the paper.

## Relevant Publications

- Lee, D., Zhang, L., Yu, Y., & Chen, W. (2024). [Deep Neural Operator Enabled Concurrent Multitask Design for Multifunctional Metamaterials under Heterogeneous Fields](https://arxiv.org/abs/2312.02403). Advanced Optical Materials (accepted).
- Lee, D. (2023). Data-Driven Inverse Design of Metamaterials: Exploring Complex Physical Fields Beyond Homogenization Assumptions (Doctoral dissertation, Northwestern University).
- Lee, D., Chen, W., Wang, L., Chan, Y. C., & Chen, W. (2023). [Data‐Driven Design for Metamaterials and Multiscale Systems: A Review](https://onlinelibrary.wiley.com/doi/10.1002/adma.202305254). Advanced Materials, 2305254.




