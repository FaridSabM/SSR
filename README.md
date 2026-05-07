# SSR

This repository provides an implementation for the paper:

Softmax Self Representation learning for Unsupervised Feature Selection

Hossein Nasserassadi, Faranges Kyanfar, Farid Saberi-Movahed, Abbas Salemi

# Description:

Self-representation (SR) models are widely used in unsupervised learning tasks such as feature selection and clustering because they effectively capture intrinsic relationships among samples and features. However, existing SR methods often rely on simple nonnegativity constraints, which are insufficient to properly regulate representation coefficients. Consequently, the learned weight matrices may become dense, unstable, and difficult to interpret.
To address these limitations, we propose a softmax-based reparameterization framework for SR learning. By applying softmax normalization to both sample and feature self-representation matrices, the proposed approach naturally enforces nonnegativity and unit-sum constraints while preventing coefficient explosion. This probabilistic formulation promotes competitive, sparse, and interpretable affinity structures, leading to more stable and reliable representations.
Another key advantage of the proposed framework is that it transforms the original constrained optimization problem into an unconstrained one, enabling efficient and stable gradient-based optimization. Based on this formulation, we develop three SR variants: Softmax SR, Softmax Mixture SR, and Softmax Bilinear SR, each equipped with an efficient iterative optimization algorithm.
Extensive experiments on four benchmark datasets demonstrate that the proposed methods consistently outperform several state-of-the-art approaches in clustering and feature selection tasks. The results confirm the effectiveness, robustness, and practical potential of the proposed softmax reparameterization strategy for improving SR model optimization.

# Citation

If you find this work useful in your research, please consider citing:

Hossein Nasserassadi, Faranges Kyanfar, Farid Saberi-Movahed, Abbas Salemi, Softmax Self Representation learning for Unsupervised Feature Selection, Journal of Mathematical Modeling, 2026.

# A quick start:

This codebase has been implemented in Python (2026). To run the project, simply execute the file main.py and follow the printed outputs in the console.
The project structure is organized as follows:

main.py – Entry point of the program.

# Contact

If you have any questions about the method, the implementation, or potential research collaborations, feel free to contact us.


Farid Saberi-Movahed
Email: f.saberimovahed@kgut.ac.ir; fdsaberi@gmail.com 
