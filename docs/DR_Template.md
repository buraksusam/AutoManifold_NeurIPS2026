# Dimensionality Reduction Algorithm Design

---

## 1 Auxiliary Tasks

This section includes a list of auxiliary tasks for an AI agent to solve, which will support the visualisation algorithm design. The input is a set of $\{\mathbf{x}_i\}_{i=1}^{n}$ data points in high-dimensional space of dimension $D$.

### 1.1 Identify Embedding Difficulty Level

Divide the $n$ data points into groups that are "hard-to-embed" (H), "moderately difficult to embed" (M), and "easy-to-embed" (E) when being mapped into a low-dimensional space with $d = 2$. Higher embedding reliability scores indicate less difficulty level to embed.

**Approach 1:** Divide data points into local patches each containing around $m = 100$ data points using a clustering algorithm, e.g., k-means clustering. Run t-SNE on each patch of data points. Calculate global and local embedding reliability scores as in Susam & Mu, *Explainable Iterative Data Visualisation Refinement via an LLM Agent* (IEEE CG&A 2026). Repeat the same for UMAP. Based on these reliability scores, group local patches into three groups of "hard-to-reduce" (H), "moderately difficult to reduce" (M), and "easy-to-reduce" (E) by setting score threshold.

**Approach 2:** Apply approach 1 with sub-sampled data points. Propagate difficulty level to nearby groups.

### 1.2 Identify Maximal Embeddable Neighborhoods

Estimate the radius of a local neighborhood that can be embedded approximately isometrically, for a set of target points.

**Approach 1:** Apply t-SNE to embed data points within a local neighbourhood of a target point, indicated by a radius in terms of a neighbour number $k$ or a distance threshold. Compute average reliability metrics for embedding neighbourhoods of a group of target points using the same radius. Grid search the radius to find the largest radius that offers reliability scores above an acceptable threshold.

**Approach 2:** Extract geometric features that can be used to estimate the radius of such maximal embeddable neighborhoods, and construct a model to estimate the radius for a group of input target points.

### 1.3 Design Local Patch Stitching Algorithm

Divide data points into local patches, assuming each local patch of data points already have a decent internal neighbor order. Ask AI agent to design patch stitching algorithms to find the relative position, rotation, reflection, and scale of the local patches so the global neighbor order is as consistent as possible. Consider both situations where patches have and do not have overlapping points. Ask the agent to find existing mathematical problem closest to this setup and re-use existing solutions. Ask the agent not to design new algorithms if not having to.

Example prompt: *"There are three local patches of data points. Within each patch, the points have fairly correct neighbour order. How to put these three local patches in the space so that all the points have fairly correct neighbour order?"*

### 1.4 Identify Impossible to Embed Points

In general, there exist a set of high-dimensional data points in $\{\mathbf{x}_i\}_{i=1}^{n}$, for which it is hard to preserve their distances with other data points. By removing these, the embedding reliability scores should significantly increase, which can be used as a metric for the AI agent to locate the points. Design an algorithm to identify these points and check its amount. It can be useful to remove these points from the plot, and visualise them separately.

---

## 2 Implementation Strategies

### 2.1 Data Feeding Strategies

Here is a list of strategies for feeding data when creating visualisation plot.

**Strategy 1:** Prepare three different visualisation plots, each by prioritising embedding accuracy for one of the three groups (H, M, and E).

**Strategy 2:** Divide data points into local patches based on maximal embeddable neighbourhoods. Embed each local patch, and then apply local patch stitching algorithms to combine.

**Strategy 3:** Incrementally optimise embeddings in the order of easy, moderate, and hard groups.

**Strategy 4:** Visualise impossible to embed points separately.

### 2.2 Fine-tune Strategies

Given the original data points $\{\mathbf{x}_i\}_{i=1}^{n}$ in high-dimensional space and the embedded data points $\{\mathbf{z}_i\}_{i=1}^{n}$ in low-dimensional space, divide the embedded data points into local patches, and calculate the reliability scores for different patches.

**Strategy 1:** Identify some local patches with particularly low scores, and directly fine-tune embedding positions of those patches to boost the scores up.

**Strategy 2:** Select local patches to directly fine-tune their embedding positions so that the overall reliability scores have the highest chance to increase.

### 2.3 Hyper-parameter Optimisation Strategy

Whenever using an algorithm, there should be hyperparameter optimisation associated with it, following the algorithm configuration optimisation strategy in Susam & Mu (IEEE CG&A 2026).
