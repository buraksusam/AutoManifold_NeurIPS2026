# Dimensionality Reduction Algorithm Design (v2)

---

## 1 Code Classification

### 1.1 Verified Deterministic Code

Verified code, written or reviewed by a human programmer, is used in different design experiments or single run of functional experiments. Once these codes are verified, they are fixed during design experiments. AI agents will be prompted explicitly to use these codes. We group such codes as synthetic data generation, score calculation, and functional algorithms. They are detailed as below.

#### 1.1.1 Synthetic Data Generation

**Manifold Construction:** Generate 3D data points drawn from 2D synthetic manifold, e.g., S-curve data, swiss roll data, or generate 2D data points drawn from 1D manifold, e.g., a wave curve.

**Surface of 3D Object:** Use the surface of 3D objects as a 2-D manifold in 3D space, and represent the surface by point clouds. The two surfaces should match before and after applying dimensionality reduction algorithm with reduced dimension set as $d = 3$.

**Black/White Image to 2D Points:** Create 2D synthetic data by treating image pixel positions as 2-dimensional points and label them by thresholding pixel values.

**Multi-cluster 2/3D Points:** Generate 2D or 3D data points containing clusters with distinct and diverse geometric shapes, with cluster number ranging from small to large.

**Multi-cluster HD Points:** Generate high-dimensional (HD) data points containing clusters drawn from probabilistic distributions of different means and variances, with cluster number ranging from small to large.

**Adding Noise:** Denote 2D/3D data points generated above by $\mathbf{X}_0$. We can add weak Gaussian noise to it, for which the noise level should not affect the main structure of the data points after applying a dimensionality reduction algorithm.

**Dimension Expansion:** For the generated data matrices $\mathbf{X}_0 \in \mathbb{R}^{n \times D_0}$ with $D_0 = 2, 3$, we can expand it to $\mathbf{X} \in \mathbb{R}^{n \times D}$ by setting the expanded dimensions as same values for all data points, e.g., $[a_{D_0+1}, a_{D_0+2}, \ldots, a_D]$ for all rows of $\mathbf{X}$. Then, we can add weak Gaussian noise to $\mathbf{X}$, slightly disturbing the data but maintaining the main 2/3D structure in data.

**Agentic Data Generation:** Ask AI to generate synthetic datasets for examining DR algorithms, being prompted by example approaches for generating data as above.
#### 1.1.2 Embedding Quality Score

Both trustworthiness $T(k)$ and continuity $C(k)$ require a pre-defined neighbourhood size. A classical way for this is to compare over ranging (from local to global) neighborhood sizes, i.e.,

$$
T = \frac{1}{k_\ell}\sum_{k=1}^{k_\ell} T(k), \qquad C = \frac{1}{k_\ell}\sum_{k=1}^{k_\ell} C(k).
$$

The targeted maximal neighborhood size ($k_\ell$) can be set differently for points from different data clusters, e.g., half of the data points for large data cluster, while larger $k_\ell$ for a small data cluster till points from nearby cluster are included to the neighbourhood.

Below is another local neighbour preservation score, which is designed to tolerate neighborhood points that are wrongly placed but still quite close to the target by introducing a neighboring relaxation strategy. It is given as

$$
P_\ell = \frac{1}{k_\ell}\sum_{k=1}^{k_\ell}\sum_{i=1}^{n}
\frac{\left|\, k_\ell\text{-NN}(\mathbf{x}_i) \cap k_\ell\text{-NN}_{\hat\delta}(\mathbf{z}_i) \,\right|}{k},
\tag{1}
$$

where relaxed near-neighbor finder is applied in the embedded space as

$$
k_\ell\text{-NN}_{\hat\delta}(\mathbf{z}_i) = \left\{ \mathbf{z}_j \;\middle|\;
\|\mathbf{z}_i - \mathbf{z}_j\|_2 \le \|\mathbf{z}_i - \mathbf{z}_i^{e_1}(k_\ell)\|_2
+ \hat\delta\, \|\mathbf{z}_i^{e_1}(k_\ell) - \mathbf{z}_i^{e_2}(k_\ell)\|_2 \right\}.
\tag{2}
$$

$\mathbf{z}_i^{e_1}(k_\ell)$ and $\mathbf{z}_i^{e_2}(k_\ell)$ denote the first and second farthest neighbors of $\mathbf{z}_i$ within the $k_\ell$-NN region in the embedded spaces, the relaxation parameter $0 \le \hat\delta < 1$ allows the placement of neighboring points outside the target neighborhood, but not too far away. When $\hat\delta = 0$, $k_\ell\text{-NN}_{\hat\delta}(\mathbf{z}_i) = k_\ell\text{-NN}(\mathbf{z}_i)$. Note that while for calculating $k_\ell\text{-NN}_{\hat\delta}(\mathbf{z}_i)$ the Euclidean distance is used as standard, for $k_\ell\text{-NN}(\mathbf{x}_i)$ other distance metrics, more suitable to the nature of the original data space could be considered. A recommended setting according to empirical experience is $\hat\delta = 0.1$. Higher values of $P_\ell$ indicate a better preservation of the local distance structure with the local region examined of a half size of the minimum cluster.

Here is another score for assessing global level of cluster structure preservation. It compares the distance orders of between data clusters in the original and embedded spaces. It is based on the Spearman's rank correlation coefficient, given as

$$
P_g = \frac{1}{C}\sum_{c=1}^{C} \operatorname{spearman}(\boldsymbol{\pi}_c, \hat{\boldsymbol{\pi}}_c)
= 1 - \frac{6}{C}\sum_{c=1}^{C} \frac{(\boldsymbol{\pi}_c - \hat{\boldsymbol{\pi}}_c)^2}{n(n^2-1)},
\tag{3}
$$

where the $(C-1)$-dimensional vector $\boldsymbol{\pi}_c$ stores the cluster rankings in terms of the closeness between the data cluster $c$ and the remaining $C-1$ clusters in the original space $\mathbb{R}^D$, while $\hat{\boldsymbol{\pi}}_c$ is defined in the same way but in the embedded space $\mathbb{R}^d$. To be consistent with the cluster embedding process, a Euclidean distance-based average-linkage distance is used for obtaining $\boldsymbol{\pi}_c$ and $\hat{\boldsymbol{\pi}}_c$.
#### 1.1.3 Functional Algorithm Code (FAC)

These correspond to codes that implement a specific existing dimensionality reduction algorithm, and the algorithms serving auxiliary tasks. The emphasis here is that these algorithms should be verified by us, through experiments using synthetic and real-world datasets. These algorithms should not change during agentic AI design phase.

### 1.2 Dynamic Codes

These codes reflect new design provided by the visualisation algorithm design pipeline. They differ across design runs.

---

## 2 Auxiliary Tasks with Code Returned

This section includes a list of auxiliary tasks for an AI agent to solve, which will support the visualisation algorithm design. The input is a set of $\{\mathbf{x}_i\}_{i=1}^{n}$ data points in high-dimensional space of dimension $D$. Each auxiliary task should be solved by the agent separately, returning code of a final algorithm accompanied by a fixed hyperparameter choosing mechanism when it fits.

### 2.1 Identify Embedding Difficulty Level

Divide the $n$ data points into groups that are "hard-to-embed" (H), "moderately difficult to embed" (M), and "easy-to-embed" (E) when being mapped into a low-dimensional space with $d = 2$. Higher embedding reliability scores indicate less difficulty level to embed.

**Detect-HME Algorithm:** Divide data points into local patches each containing around $m = 100$ data points using a clustering algorithm, e.g., k-means clustering. Run t-SNE on each patch of data points. Calculate global and local embedding reliability scores as in Susam & Mu, *Explainable Iterative Data Visualisation Refinement via an LLM Agent* (IEEE CG&A 2026). Repeat the same for UMAP. Based on these reliability scores, group local patches into three groups of "hard-to-reduce" (H), "moderately difficult to reduce" (M), and "easy-to-reduce" (E) by setting score threshold. This Detect-HME algorithm should exist as a piece of verified code, taking a dataset as input, returning H, M and E as output.

**Hyper-parameter Search:** Hyper-parameters of the Detect-HME algorithm include (1) which reliability scores to use, (2) what score combination to use for calculating the final score to threshold, and (3) what score threshold to use. Ask an AI agent to come up with an explicit score calculation and threshold determination mechanism, reflected as verified code.

**Evaluation and Datasets:** Synthesize multiple 3-D datasets drawn from different 2D manifolds containing different numbers of points. The dimensionality reduction algorithm can at most embed one manifold correctly. By changing manifold shapes and point numbers, we can engineer groups of points that are "hard-to-reduce" (H), "moderately difficult to reduce" (M), and "easy-to-reduce" (E). We can also generate high-dimensional data points of different dimensions, and expand their dimension to the same number $D$ so that they sit in different subspaces of the $D$-dimensional space. The dimensionality reduction algorithm can find at most one correct subspace to embed. By adding noise with different strengths and changing point numbers in different subspaces, we can engineer H, M and E groups of points. For a given dataset with annotated H, M and E patches, performance of Detect-HME can then be evaluated. Datasets and evaluation code must be explicitly defined and verified by a human.

**AI Agentic Data Generation:** Ask AI to synthesize datasets with H, M, E annotation, being prompted by examples ways of generating datasets with H, M, and E annotation.

**AI Agentic Design:** Use Detect-HME algorithm as a starting algorithm, ask AI to design a better algorithm than Detect-HME, i.e., to perform better than Detect-HME on the provided datasets with HME annotations.

### 2.2 Identify Maximal Embeddable Neighborhoods

Estimate the radius of a local neighborhood that can be embedded approximately isometrically, for a patch of points.

**Embeddable Radius Algorithm:** Divide data points into local patches each containing around $m = 100$ data points using a clustering algorithm, e.g., k-means clustering. Apply a dimensionality reduction algorithm, e.g., t-SNE, UMAP, or PaCMAP, to embed data points within a local neighbourhood of a target point from a local patch, indicated by a radius in terms of a neighbour number $k$ or a distance threshold. Compute average reliability scores for embedding neighbourhoods of all points from the patch using the same radius. Grid search the radius to find the largest radius that offers reliability scores above an acceptable threshold. Equivalently, can also grid search the smallest radius with significant reliability score drop. This algorithm should exist as a piece of verified code, taking a dataset and a dimensionality reduction algorithm as input, returning a set of radius values corresponding to different local patches.

**Hyper-parameter Search:** Hyper-parameters of the embeddable radius algorithm include (1) which local reliability scores to use, (2) what score combination to use for checking the final score, and (3) what threshold to use for deciding an acceptable score (maximum radius) or a significant score drop (minimum radius). Ask an AI agent to come up with an explicit score calculation and threshold selection mechanism, reflected as verified code.

**Evaluation and Datasets:** Synthesize multiple clusters of 2-D data points in the same space with different region coverage ranging from small to large, and expand these points to 3-D by adding a fixed value to the third dimension (might add weak noise), to make sure these clusters are all embeddable when mapping from $D = 3$ space to $d = 2$ space. Then, construct sets of 3D points on different 2D manifolds and sets of 3D points that cannot be embedded to 2D space at all. Mix these with embeddable clusters. This results in engineered datasets with different local patches, of which some are embeddable while some are not. Test algorithms using such datasets to match the computed values of radius. Datasets and evaluation code must be explicitly defined and verified by a human.

**AI Agentic Data Generation:** Ask AI to synthesize datasets containing both embeddable patches and non-embeddable patches, being prompted by examples ways of generating such datasets.

**AI Agentic Design:** Use embeddable radius algorithm as a starting algorithm, ask AI to design a machine learning algorithm to extract geometric features that can be used to estimate the radius of such maximal embeddable neighborhoods, and construct a model to estimate the radius for a given patch, and evaluate the designed ML algorithm with datasets.

### 2.3 Design Local Patch Stitching Algorithm

**Algorithm Goal:** Divide data points into local patches, assuming each local patch of data points already have a decent internal neighbor order. Ask AI agent to design patch stitching algorithms to find the relative position, rotation, reflection, and scale of the local patches so the global neighbor order is as consistent as possible. Consider both situations where patches have and do not have overlapping points. Ask the agent to find existing mathematical problem closest to this setup and re-use existing solutions. Ask the agent not to design new algorithms if not having to.

Example prompt: *"There are three local patches of data points. Within each patch, the points have fairly correct neighbour order. How to put these three local patches in the space so that all the points have fairly correct neighbour order?"*

**Evaluation and AI Target:** Take a set of data points, and run t-SNE on all points. Then divide data points to e.g., $N = 10, 20, \ldots, 100$ patches. Use t-SNE on each patch, and use the stitching algorithm to stitch together the embedded patches forming a new embedding arrangement. Compare t-SNE results with t-SNE+stitching results on local and global aspects. The AI agent is asked to maximize local score maintaining good global level (same design goal as in Section 3).

### 2.4 Identify Impossible to Embed Points

**Algorithm Goal:** In general, there exist a set of high-dimensional data points in $\{\mathbf{x}_i\}_{i=1}^{n}$, for which it is hard to preserve their distances with other data points. By removing these, the embedding reliability scores should significantly increase, which can be used as a metric for the AI agent to locate the points. Design an algorithm to identify these points and check its amount. It can be useful to remove these points from the plot, and visualize them separately.

**Mathematical Formulation:** Find the minimal set containing no more than $n^*$ data points under the constraint of $n^* \le \rho n$ ($\rho$ is a percentage of original point number allowed to remove), to remove in order to maximize the reliability score. Ask AI to design a fast algorithm for this.

**Evaluation:** Simply compare the reliability score before and after removing the points.

---

## 3 Design Target

We need a clearly defined metric to drive the embedding algorithm design to improve over. In general, local neighbor preservation accuracy is the only metric worth to quantitatively aim at. Because the score difference means actually accuracy improvement in preserving original data information. Global structure, e.g., cluster shapes, cluster separability, inter-cluster distance order, is more for visual observation, which is more suitable to be assessed by some satisfactory level, e.g., good, ok, or bad, rather than a real-valued score. Here is a list of options for quantified design target.

- Ask the agent to take an overall consideration of all local and global scores. This is not recommended, because strong algorithms could perform similarly well across all scores on average, for which some score difference might not indicate much visual difference and accuracy difference.
- Ask the agent to aim at maximizing local score only, for which the examined local neighbourhood is chosen appropriately.
- Define an acceptable range of global scores, or even an acceptable visual looking. Within this acceptable range, maximize only local scores.

---

## 4 Data and Optimization Pipelines

In general, we should exploit AI agents by taking advantage of their search capacity in a design space and a heuristic process pipeline. We can work at providing choice ranges and design boundaries for AI agents to explore, for which a design success is clearly quantified as discussed in Section 3. We exemplify below design/process choices on different design aspects.

### 4.1 Single vs. Multiple Plots

The AI agent can choose between producing (1) one single visualisation for all data points and (2) multiple visualisation (e.g., no more than 3) for groups of data points. Evaluation of multiple visualizations would be averaged reliability scores over the multiple visualization plots. Here are some example strategies.

**Strategy 1:** Prepare three different visualisation plots, each by prioritising embedding accuracy for one of the three groups (H, M, and E).

**Strategy 2:** Visualise impossible to embed points separately.

**Strategy 3:** Ask AI to come up with a way to divide points to produce multiple visualizations to boost averaged scores.

### 4.2 Data Feeding and Fine-tuning

Given the original data points $\{\mathbf{x}_i\}_{i=1}^{n}$ in high-dimensional space and the embedded data points $\{\mathbf{z}_i\}_{i=1}^{n}$ in low-dimensional space, the algorithm has choices on how to use data in training (optimization). Here are some example strategies.

**Strategy 1:** Divide the data to easy, moderate, and hard groups. Optimize embeddings for easy group first. Fix easy embeddings, and optimize moderate embeddings. Fix easy and moderate embedding, and optimize hard embeddings. Finally, fine tune all embeddings.

**Strategy 2:** Divide data points into local patches based on maximal embeddable neighbourhoods. Optimize embeddings for data points from each local patch, and then apply local patch stitching algorithms to compute new embeddings for all points.

**Strategy 3:** Divide data into local patches. Ask AI agent to identify some local patches with particularly low scores, and directly fine-tune embedding positions of those patches to boost the scores up.

**Strategy 4:** Ask AI agent to select local patches or set of points to directly fine-tune their embedding positions to maximize reliability scores. This is equivalent to a combinatorial optimization problem (point index selection for score maximization).

### 4.3 Hyper-parameter Optimisation

Whenever using an algorithm, there should be hyperparameter optimisation associated with it, following the algorithm configuration optimisation strategy in Susam & Mu (IEEE CG&A 2026).

---

## 5 DR Algorithm Design

The purpose is to design a t-SNE and UMAP like algorithm, that can be used as a base algorithm supporting pipeline choices in Section 4.

**Approach 1:** Unconstrained design (no component guidance).

**Approach 2:** Provide algorithm component options, of which some combinations have to result in t-SNE, UMAP and PaCMAP. Do not include spectral embedding and PCA like global control within the algorithm component design, but allow to use their output as initialization of embeddings.
