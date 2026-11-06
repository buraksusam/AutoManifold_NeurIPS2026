"""
Compute engines — the non-LLM workhorses of the pipeline.

These correspond to the three engines in the paper's Figure 1:
  1. DRVisualizationEngine  — computes 2D embeddings + scatter plots
  2. QuantitativeAnalystEngine — computes reliability metrics (T, C, S, ρs, R, Silhouette, LOF)
  3. ClusteringEngine — computes hierarchical dendrograms in Newick format
"""
