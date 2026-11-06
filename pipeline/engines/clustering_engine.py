"""
Clustering Engine

Produces hierarchical clustering trees (dendrograms) using UPGMA and exports
them in Newick format — for both the original high-dimensional data and the
2D embeddings.  This allows the LLM agent to assess structural agreement
between HD and LD representations (paper §"WHAT/HOW TO INSPECT?").

Extended with:
  • Branch-length Newick strings (distances encoded)
  • Linkage matrix retention for plotting
  • HD↔LD hierarchy diff: topological changes, subtree rearrangements,
    per-class merge-order deltas
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree, leaves_list, cophenet
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


class ClusteringEngine:
    """Compute UPGMA hierarchical clustering and produce Newick strings."""

    # ------------------------------------------------------------------
    # Public API — dendrograms with full metadata
    # ------------------------------------------------------------------

    def compute_dendrograms(
        self,
        hd_data: np.ndarray,
        ld_data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        hd_label: str = "HD",
        ld_label: str = "2D",
    ) -> dict:
        """
        Compute HD and LD dendrograms and return them in the paper's schema.

        Returns a dict with hierarchy_hd, hierarchy_2d each containing:
          - newick: Newick string with branch lengths
          - dimension: label
          - linkage: the scipy linkage matrix Z
          - leaf_labels: ordered labels for leaves
          - leaf_order: leaf ordering from the linkage
        """
        if labels is not None and len(np.unique(labels)) >= 2:
            hd_result = self._dendrogram_on_centroids(hd_data, labels)
            ld_result = self._dendrogram_on_centroids(ld_data, labels)
        else:
            # Subsample to keep linkage tractable
            n = min(500, len(hd_data))
            idx = np.random.choice(len(hd_data), n, replace=False)
            hd_result = self._dendrogram_on_points(hd_data[idx])
            ld_result = self._dendrogram_on_points(ld_data[idx])

        hd_result["dimension"] = hd_label
        ld_result["dimension"] = ld_label

        return {
            "hierarchy_hd": hd_result,
            "hierarchy_2d": ld_result,
        }

    def compute_hierarchy_diff(
        self,
        dendrograms: dict,
        label_names: Optional[Dict[int, str]] = None,
        dr_method: str | None = None,
    ) -> dict:
        """
        Compare HD and LD dendrograms and return a structured diff.

        Returns dict with:
          - hd_newick, ld_newick: the Newick strings
          - leaf_order_hd, leaf_order_ld: ordered leaf labels
          - merge_order_hd, merge_order_ld: sequence of merges (pairs of labels)
          - topology_changes: list of dicts describing specific changes
          - robinson_foulds_distance: normalised RF distance (0=identical, 1=maximally different)
          - cophenetic_correlation: correlation of HD vs LD cophenetic distances
          - summary: human-readable diff string for LLM prompts
        """
        hd = dendrograms.get("hierarchy_hd", {})
        ld = dendrograms.get("hierarchy_2d", {})

        hd_Z = hd.get("linkage")
        ld_Z = ld.get("linkage")
        hd_labels = hd.get("leaf_labels", [])
        ld_labels = ld.get("leaf_labels", [])

        if hd_Z is None or ld_Z is None:
            return {"summary": "No linkage data available for comparison."}

        # --- Merge order extraction ---
        hd_merges = self._extract_merge_order(hd_Z, hd_labels)
        ld_merges = self._extract_merge_order(ld_Z, ld_labels)

        # --- Leaf order comparison ---
        hd_leaf_order = self._get_leaf_order(hd_Z, hd_labels)
        ld_leaf_order = self._get_leaf_order(ld_Z, ld_labels)

        # --- Robinson-Foulds-like distance on bipartitions ---
        rf_dist = self._robinson_foulds(hd_Z, ld_Z, hd_labels)

        # --- Cophenetic correlation ---
        hd_coph = cophenet(hd_Z)
        ld_coph = cophenet(ld_Z)
        if len(hd_coph) == len(ld_coph) and len(hd_coph) > 2:
            coph_corr = float(spearmanr(hd_coph, ld_coph).correlation)
        else:
            coph_corr = None

        # --- Topology changes: find specific merge-order differences ---
        topology_changes = self._find_topology_changes(
            hd_merges, ld_merges, hd_Z, ld_Z, hd_labels, label_names,
        )

        # --- Build human-readable summary ---
        summary_lines = []
        summary_lines.append("Hierarchy Comparison (HD vs 2D):")
        summary_lines.append(f"  HD leaf order:  {' → '.join(str(l) for l in hd_leaf_order)}")
        summary_lines.append(f"  2D leaf order:  {' → '.join(str(l) for l in ld_leaf_order)}")
        summary_lines.append(f"  Robinson-Foulds distance: {rf_dist:.3f} (0=identical, 1=max different)")
        if coph_corr is not None:
            summary_lines.append(f"  Cophenetic correlation: {coph_corr:.4f}")
        summary_lines.append(f"  HD merge order: {self._format_merges(hd_merges)}")
        summary_lines.append(f"  2D merge order: {self._format_merges(ld_merges)}")
        if topology_changes:
            summary_lines.append(f"  Topology changes ({len(topology_changes)}):")
            for tc in topology_changes[:8]:
                summary_lines.append(f"    • {tc['description']}")

        # Add DR-method-aware calibration note
        method = (dr_method or "").lower()
        if method in ("tsne", "t-sne"):
            summary_lines.append("")
            summary_lines.append("  ⚠ CALIBRATION NOTE (t-SNE): t-SNE is a LOCAL method that")
            summary_lines.append("  fundamentally does not preserve global hierarchical relationships.")
            summary_lines.append("  Negative cophenetic correlation and high RF distance are EXPECTED")
            summary_lines.append("  and NORMAL for t-SNE. These hierarchy metrics should NOT be used")
            summary_lines.append("  to penalise the quality score. Instead, use them as observational")
            summary_lines.append("  context to understand WHICH relationships were altered, not as")
            summary_lines.append("  evidence of poor embedding quality.")
        elif method in ("umap",):
            summary_lines.append("")
            summary_lines.append("  ⚠ CALIBRATION NOTE (UMAP): UMAP partially preserves global")
            summary_lines.append("  structure but hierarchy fidelity is still limited. Moderate RF")
            summary_lines.append("  distance and low cophenetic correlation are typical. Treat these")
            summary_lines.append("  as informational context, not primary optimization targets.")
        elif method in ("pacmap",):
            summary_lines.append("")
            summary_lines.append("  ⚠ CALIBRATION NOTE (PaCMAP): PaCMAP balances local and global")
            summary_lines.append("  structure. Some hierarchy distortion is normal. These metrics are")
            summary_lines.append("  informational context, not primary optimization targets.")

        return {
            "hd_newick": hd.get("newick", "N/A"),
            "ld_newick": ld.get("newick", "N/A"),
            "leaf_order_hd": hd_leaf_order,
            "leaf_order_ld": ld_leaf_order,
            "merge_order_hd": hd_merges,
            "merge_order_ld": ld_merges,
            "topology_changes": topology_changes,
            "robinson_foulds_distance": round(rf_dist, 4),
            "cophenetic_correlation": round(coph_corr, 4) if coph_corr is not None else None,
            "summary": "\n".join(summary_lines),
        }

    # ------------------------------------------------------------------
    # Internals — dendrogram construction
    # ------------------------------------------------------------------

    def _dendrogram_on_centroids(
        self, data: np.ndarray, labels: np.ndarray
    ) -> dict:
        """Build UPGMA tree on per-label centroids. Returns dict with newick, linkage, leaf_labels."""
        unique_labels = np.unique(labels)
        centroids = np.array(
            [data[labels == lab].mean(axis=0) for lab in unique_labels]
        )
        dists = pdist(centroids, metric="euclidean")
        Z = linkage(dists, method="average")  # UPGMA
        tree = to_tree(Z)
        label_list = [str(lab) for lab in unique_labels]
        label_map = {i: label_list[i] for i in range(len(unique_labels))}
        newick = self._tree_to_newick(tree, label_map) + ";"

        return {
            "newick": newick,
            "linkage": Z,
            "leaf_labels": label_list,
            "leaf_order": self._get_leaf_order(Z, label_list),
        }

    def _dendrogram_on_points(self, data: np.ndarray) -> dict:
        """Build UPGMA tree on raw data points."""
        dists = pdist(data, metric="euclidean")
        Z = linkage(dists, method="average")
        tree = to_tree(Z)
        label_list = [str(i) for i in range(len(data))]
        label_map = {i: label_list[i] for i in range(len(data))}
        newick = self._tree_to_newick(tree, label_map) + ";"

        return {
            "newick": newick,
            "linkage": Z,
            "leaf_labels": label_list,
            "leaf_order": self._get_leaf_order(Z, label_list),
        }

    def _tree_to_newick(self, node, label_map: dict) -> str:
        """Recursively convert a scipy ClusterNode to a Newick string with branch lengths."""
        if node.is_leaf():
            return label_map.get(node.id, str(node.id))
        left = node.get_left()
        right = node.get_right()
        left_str = self._tree_to_newick(left, label_map)
        right_str = self._tree_to_newick(right, label_map)
        # Add branch lengths (distance from child to this node)
        left_bl = round(node.dist - (left.dist if not left.is_leaf() else 0), 4)
        right_bl = round(node.dist - (right.dist if not right.is_leaf() else 0), 4)
        return f"({left_str}:{left_bl},{right_str}:{right_bl})"

    # ------------------------------------------------------------------
    # Internals — merge order + leaf order extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_merge_order(Z: np.ndarray, leaf_labels: list) -> list:
        """
        Extract merge order from linkage matrix Z.
        Returns list of (cluster_a, cluster_b, distance) in merge order.
        """
        n = len(leaf_labels)
        cluster_map = {i: leaf_labels[i] for i in range(n)}
        merges = []
        for i, row in enumerate(Z):
            a_idx, b_idx, dist = int(row[0]), int(row[1]), row[2]
            a_name = cluster_map.get(a_idx, f"?{a_idx}")
            b_name = cluster_map.get(b_idx, f"?{b_idx}")
            merged_name = f"({a_name}+{b_name})"
            cluster_map[n + i] = merged_name
            merges.append((a_name, b_name, round(float(dist), 4)))
        return merges

    @staticmethod
    def _get_leaf_order(Z: np.ndarray, leaf_labels: list) -> list:
        """Return leaf labels in dendrogram left-to-right order."""
        order_indices = leaves_list(Z)
        return [leaf_labels[i] for i in order_indices if i < len(leaf_labels)]

    @staticmethod
    def _format_merges(merges: list) -> str:
        """Format merge list as a readable string."""
        parts = []
        for a, b, d in merges:
            parts.append(f"[{a} ⊕ {b} @ d={d}]")
        return " → ".join(parts)

    # ------------------------------------------------------------------
    # Internals — Robinson-Foulds distance
    # ------------------------------------------------------------------

    @staticmethod
    def _get_bipartitions(Z: np.ndarray, n_leaves: int) -> set:
        """Extract all bipartitions (splits) from a linkage matrix."""
        cluster_members = {i: frozenset([i]) for i in range(n_leaves)}
        splits = set()
        for i, row in enumerate(Z):
            a, b = int(row[0]), int(row[1])
            members_a = cluster_members[a]
            members_b = cluster_members[b]
            merged = members_a | members_b
            cluster_members[n_leaves + i] = merged
            if len(members_a) > 0 and len(members_a) < n_leaves:
                smaller = members_a if len(members_a) <= n_leaves // 2 else frozenset(range(n_leaves)) - members_a
                splits.add(smaller)
        return splits

    def _robinson_foulds(
        self, Z_hd: np.ndarray, Z_ld: np.ndarray, leaf_labels: list
    ) -> float:
        """
        Compute normalised Robinson-Foulds distance between HD and LD trees.
        Returns float in [0, 1].
        """
        n = len(leaf_labels)
        if n < 3:
            return 0.0
        hd_splits = self._get_bipartitions(Z_hd, n)
        ld_splits = self._get_bipartitions(Z_ld, n)
        sym_diff = len(hd_splits ^ ld_splits)
        max_possible = len(hd_splits) + len(ld_splits)
        if max_possible == 0:
            return 0.0
        return sym_diff / max_possible

    # ------------------------------------------------------------------
    # Internals — topology change detection
    # ------------------------------------------------------------------

    def _find_topology_changes(
        self,
        hd_merges: list,
        ld_merges: list,
        Z_hd: np.ndarray,
        Z_ld: np.ndarray,
        leaf_labels: list,
        label_names: Optional[Dict[int, str]] = None,
    ) -> list:
        """
        Compare HD and LD merge sequences to find specific topological changes.
        Returns a list of dicts describing each change.
        """
        changes = []

        def _name(lab_str):
            if label_names:
                try:
                    return label_names.get(int(lab_str), lab_str)
                except (ValueError, TypeError):
                    return lab_str
            return lab_str

        # 1. Closest pair changed
        if hd_merges and ld_merges:
            hd_first = (hd_merges[0][0], hd_merges[0][1])
            ld_first = (ld_merges[0][0], ld_merges[0][1])
            hd_set = {hd_first[0], hd_first[1]}
            ld_set = {ld_first[0], ld_first[1]}
            if hd_set != ld_set:
                changes.append({
                    "type": "closest_pair_changed",
                    "hd_closest": [_name(hd_first[0]), _name(hd_first[1])],
                    "ld_closest": [_name(ld_first[0]), _name(ld_first[1])],
                    "description": (
                        f"Closest pair changed: HD has {_name(hd_first[0])}↔{_name(hd_first[1])} "
                        f"but 2D has {_name(ld_first[0])}↔{_name(ld_first[1])}"
                    ),
                    "severity": "high",
                })

        # 2. Nearest-neighbour changes per leaf
        def _first_merge_partner(merges, leaf_set):
            partners = {}
            merged_set = set()
            for a, b, d in merges:
                if a in leaf_set and b in leaf_set:
                    if a not in merged_set:
                        partners[a] = b
                    if b not in merged_set:
                        partners[b] = a
                    merged_set.add(a)
                    merged_set.add(b)
            return partners

        leaf_set = set(leaf_labels)
        hd_partners = _first_merge_partner(hd_merges, leaf_set)
        ld_partners = _first_merge_partner(ld_merges, leaf_set)

        for leaf in leaf_labels:
            hd_p = hd_partners.get(leaf)
            ld_p = ld_partners.get(leaf)
            if hd_p and ld_p and hd_p != ld_p:
                changes.append({
                    "type": "nearest_neighbour_changed",
                    "class": _name(leaf),
                    "hd_nearest": _name(hd_p),
                    "ld_nearest": _name(ld_p),
                    "description": (
                        f"Class {_name(leaf)}: nearest hierarchical neighbour "
                        f"changed from {_name(hd_p)} (HD) to {_name(ld_p)} (2D)"
                    ),
                    "severity": "medium",
                })

        # 3. Merge-order inversions
        def _merge_rank(merges, leaf_labels_set):
            n_l = len(leaf_labels_set)
            cluster_members = {l: {l} for l in leaf_labels_set}
            pair_rank = {}
            for step, (a, b, d) in enumerate(merges):
                members_a = cluster_members.get(a, {a})
                members_b = cluster_members.get(b, {b})
                merged = members_a | members_b
                new_name = f"({a}+{b})"
                cluster_members[new_name] = merged
                for la in members_a:
                    for lb in members_b:
                        key = tuple(sorted([la, lb]))
                        if key not in pair_rank:
                            pair_rank[key] = step
            return pair_rank

        hd_ranks = _merge_rank(hd_merges, leaf_labels)
        ld_ranks = _merge_rank(ld_merges, leaf_labels)

        for pair, hd_rank in hd_ranks.items():
            ld_rank = ld_ranks.get(pair)
            if ld_rank is not None:
                rank_diff = ld_rank - hd_rank
                if abs(rank_diff) >= 2:
                    a, b = pair
                    direction = "later" if rank_diff > 0 else "earlier"
                    changes.append({
                        "type": "merge_order_inversion",
                        "classes": [_name(a), _name(b)],
                        "hd_merge_rank": hd_rank,
                        "ld_merge_rank": ld_rank,
                        "description": (
                            f"Classes {_name(a)} & {_name(b)} merge {direction} "
                            f"in 2D (rank {ld_rank}) vs HD (rank {hd_rank}) — "
                            f"Δ={rank_diff:+d} steps"
                        ),
                        "severity": "high" if abs(rank_diff) >= 3 else "medium",
                    })

        severity_order = {"high": 0, "medium": 1, "low": 2}
        changes.sort(key=lambda c: severity_order.get(c.get("severity", "low"), 2))
        return changes
