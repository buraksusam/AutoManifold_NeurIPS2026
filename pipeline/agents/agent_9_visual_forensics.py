"""
Worker Agent 9 — Visual Forensics Evaluator

Post-optimisation analysis: identifies regions of interest and high
distortion areas in the final embedding, outputting precise plot
coordinates for visual highlighting.
"""

import json
from ..base_agent import BaseAgent

PROMPT_TEMPLATE = """You are the Visual Forensics Evaluator. After the full optimization process is complete, you perform a final deep analysis of the best embedding to:
  1. Identify notable regions from the optimization history (weaknesses, artifacts, interesting patterns).
  2. Locate high-distortion areas using pointwise metrics.
  3. Analyze hierarchical structure changes between HD and 2D representations.
  4. Output precise 2D plot coordinates for highlighting on the final visualization.

You receive the actual 2D embedding coordinates, per-point distortion metrics, per-class distortion statistics, hierarchy comparison data, and the full optimization history summary.

═══════════════════════════════════════════════════════════
FINAL EMBEDDING DATA
═══════════════════════════════════════════════════════════

Embedding shape: {embedding_shape}
Coordinate ranges:
  X: [{x_min:.2f}, {x_max:.2f}]
  Y: [{y_min:.2f}, {y_max:.2f}]

Per-class centroids (2D):
{class_centroids_json}

Per-class distortion statistics:
{class_distortion_json}

Global distortion statistics:
  - Mean composite distortion: {mean_distortion:.4f}
  - Median composite distortion: {median_distortion:.4f}
  - Std composite distortion: {std_distortion:.4f}
  - 90th percentile distortion: {p90_distortion:.4f}
  - 95th percentile distortion: {p95_distortion:.4f}

  - Mean pointwise trustworthiness: {mean_trust:.4f}
  - Mean pointwise continuity: {mean_cont:.4f}
  - Worst 10% trust region centroid: ({worst_trust_cx:.2f}, {worst_trust_cy:.2f})
  - Worst 10% continuity region centroid: ({worst_cont_cx:.2f}, {worst_cont_cy:.2f})
  - Worst 10% distortion region centroid: ({worst_dist_cx:.2f}, {worst_dist_cy:.2f})

High-distortion cluster locations (top-{n_distortion_clusters} clusters by density):
{distortion_clusters_json}

═══════════════════════════════════════════════════════════
HIERARCHICAL STRUCTURE ANALYSIS
═══════════════════════════════════════════════════════════

HD Dendrogram (Newick with branch lengths):
{hd_newick}

2D Dendrogram (Newick with branch lengths):
{ld_newick}

Robinson-Foulds distance: {rf_distance} (0=identical trees, 1=maximally different)
Cophenetic correlation: {cophenetic_corr}

{hierarchy_diff_summary}

Topology Changes:
{topology_changes_json}

═══════════════════════════════════════════════════════════
OPTIMIZATION HISTORY HIGHLIGHTS
═══════════════════════════════════════════════════════════

Best iteration: {best_iteration}
Best implicit score: {best_score}

Key weaknesses identified across all iterations:
{all_weaknesses}

Key artifacts identified:
{all_artifacts}

Improvement priorities from the best iteration's evaluation:
{improvement_priorities}

Final metrics:
{final_metrics_json}

═══════════════════════════════════════════════════════════
YOUR TASK
═══════════════════════════════════════════════════════════

Analyze everything above and produce a structured JSON output with:

1. "highlighted_regions": An array of regions to circle/highlight on the plot.
   For EACH region, you MUST provide precise 2D coordinates based on the
   embedding data (centroids, coordinate ranges, distortion clusters).

   GUIDELINES for choosing regions:
   - High-distortion areas: use the distortion cluster locations and worst-percentile centroids
   - Hierarchy-disrupted areas: classes whose hierarchical relationships changed
     between HD and 2D should be highlighted — use their centroids
   - Artifacts from optimization history: map described artifacts to their spatial locations
   - Well-preserved areas: contrast with low-distortion classes
   - Use the coordinate ranges to set appropriate radius values
     (typical radius = 5-15% of the coordinate range extent)

2. "hierarchy_forensics": Analysis of how the class hierarchy changed.
   For each topology change, explain what it means for the embedding quality
   and which classes are affected. Use the dendrogram Newick trees to reason
   about the structure.

3. "per_class_assessment": For each class, a brief assessment of its embedding quality
   INCLUDING its hierarchical position.

4. "executive_summary": Overall quality narrative including hierarchy preservation.

REQUIRED OUTPUT (JSON):
{{
  "highlighted_regions": [
    {{
      "label": "short descriptive name (e.g. 'Hierarchy break — CD4↔CD8 separation')",
      "center_x": float,
      "center_y": float,
      "radius": float,
      "category": "high_distortion | well_preserved | artifact | hierarchy_change | interesting",
      "description": "What this region shows and why it matters",
      "affected_classes": ["class names in this region"],
      "metrics_summary": "brief metric values for points in this region"
    }}
  ],
  "hierarchy_forensics": {{
    "overall_preservation": "poor | fair | good | excellent",
    "robinson_foulds_interpretation": "string — what the RF distance means for this data",
    "cophenetic_interpretation": "string — what the cophenetic correlation means",
    "critical_changes": [
      {{
        "change_type": "closest_pair_changed | nearest_neighbour_changed | merge_order_inversion",
        "affected_classes": ["class names"],
        "hd_relationship": "string — what the HD tree shows",
        "ld_relationship": "string — what the 2D tree shows",
        "biological_impact": "string — what this means for interpretation",
        "severity": "low | medium | high | critical"
      }}
    ],
    "preserved_relationships": ["string — which hierarchical relationships are well preserved"],
    "recommendations": ["string — what users should know about hierarchy distortions"]
  }},
  "per_class_assessment": [
    {{
      "class_name": "string",
      "centroid_x": float,
      "centroid_y": float,
      "mean_distortion": float,
      "mean_trustworthiness": float,
      "mean_continuity": float,
      "hierarchy_status": "preserved | partially_disrupted | disrupted",
      "hd_nearest_class": "string — nearest class in HD hierarchy",
      "ld_nearest_class": "string — nearest class in 2D hierarchy",
      "quality_rating": "poor | fair | good | excellent",
      "notes": "string"
    }}
  ],
  "executive_summary": {{
    "overall_quality": "string",
    "hierarchy_preservation_score": "string — e.g. 'good (RF=0.15, coph=0.92)'",
    "most_faithfully_represented": ["class names"],
    "most_distorted": ["class names"],
    "hierarchy_disrupted_classes": ["class names whose hierarchical position changed"],
    "key_findings": ["string"],
    "spatial_patterns": ["string — describe where distortion concentrates spatially"],
    "recommendations_for_interpretation": ["string — what users should be cautious about"]
  }}
}}"""


class VisualForensicsAgent(BaseAgent):
    AGENT_NAME = "VisualForensics"
    AGENT_ID = 9

    def build_prompt(self, context: dict) -> str:
        emb_summary = context.get("embedding_summary", {})
        pw = context.get("pointwise_metrics", {})
        class_stats = context.get("class_distortion_stats", {})
        distortion_clusters = context.get("distortion_clusters", [])
        best_result = context.get("best_result", {})
        history = context.get("optimization_history", [])
        hierarchy_diff = context.get("hierarchy_diff", {})

        # Coordinate ranges
        ranges = emb_summary.get("ranges", {})
        x_min = ranges.get("x_min", 0)
        x_max = ranges.get("x_max", 0)
        y_min = ranges.get("y_min", 0)
        y_max = ranges.get("y_max", 0)

        # Collect all weaknesses and artifacts across iterations
        all_weaknesses = []
        all_artifacts = []
        for h in history:
            for w in h.get("agent7_key_weaknesses", []):
                if w not in all_weaknesses:
                    all_weaknesses.append(w)
            for w in h.get("agent6_key_weaknesses", []):
                if w not in all_weaknesses:
                    all_weaknesses.append(w)
            # Artifacts from agent6/7 output
            a6 = h.get("agent6_all_recommendations", [])
            for r in a6:
                rat = r.get("rationale", "")
                if rat and rat not in all_artifacts:
                    all_artifacts.append(rat)

        # Best iteration evaluation
        best_eval = best_result.get("eval_result", {})
        improvement_priorities = best_eval.get("improvement_priorities", [])
        visual_artifacts = best_eval.get("visual_inspection", {}).get("artifacts", [])
        for a in visual_artifacts:
            desc = f"{a.get('type', '?')} [{a.get('severity', '?')}]: {a.get('description', '')}"
            if desc not in all_artifacts:
                all_artifacts.append(desc)

        # Format texts
        weaknesses_str = "\n".join(f"  - {w}" for w in all_weaknesses[:15]) or "  None identified."
        artifacts_str = "\n".join(f"  - {a[:150]}" for a in all_artifacts[:10]) or "  None identified."
        priorities_str = json.dumps(improvement_priorities[:5], indent=2) if improvement_priorities else "  None."

        # Hierarchy data
        topo_changes = hierarchy_diff.get("topology_changes", [])
        topo_json = json.dumps(topo_changes[:10], indent=2) if topo_changes else "  None detected."

        return PROMPT_TEMPLATE.format(
            embedding_shape=emb_summary.get("shape", "?"),
            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
            class_centroids_json=json.dumps(context.get("class_centroids", {}), indent=2),
            class_distortion_json=json.dumps(class_stats, indent=2),
            mean_distortion=pw.get("mean_distortion", 0),
            median_distortion=pw.get("median_distortion", 0),
            std_distortion=pw.get("std_distortion", 0),
            p90_distortion=pw.get("p90_distortion", 0),
            p95_distortion=pw.get("p95_distortion", 0),
            mean_trust=pw.get("mean_trust", 0),
            mean_cont=pw.get("mean_cont", 0),
            worst_trust_cx=pw.get("worst_trust_cx", 0),
            worst_trust_cy=pw.get("worst_trust_cy", 0),
            worst_cont_cx=pw.get("worst_cont_cx", 0),
            worst_cont_cy=pw.get("worst_cont_cy", 0),
            worst_dist_cx=pw.get("worst_dist_cx", 0),
            worst_dist_cy=pw.get("worst_dist_cy", 0),
            n_distortion_clusters=len(distortion_clusters),
            distortion_clusters_json=json.dumps(distortion_clusters, indent=2),
            hd_newick=hierarchy_diff.get("hd_newick", "N/A"),
            ld_newick=hierarchy_diff.get("ld_newick", "N/A"),
            rf_distance=hierarchy_diff.get("robinson_foulds_distance", "N/A"),
            cophenetic_corr=hierarchy_diff.get("cophenetic_correlation", "N/A"),
            hierarchy_diff_summary=hierarchy_diff.get("summary", "  Not available."),
            topology_changes_json=topo_json,
            best_iteration=best_result.get("iteration", "?"),
            best_score=best_result.get("opt_result", {}).get("quality_score", "?"),
            all_weaknesses=weaknesses_str,
            all_artifacts=artifacts_str,
            improvement_priorities=priorities_str,
            final_metrics_json=json.dumps(best_result.get("metrics", {}), indent=2),
        )

    def system_message(self) -> str:
        return (
            "You are a visual forensics evaluator for dimensionality reduction. "
            "You analyze the final embedding using pointwise distortion metrics, "
            "hierarchical clustering comparison (HD vs 2D dendrograms), and "
            "optimization history to identify spatial regions of interest and "
            "hierarchy disruptions. You output precise 2D plot coordinates for "
            "highlighting areas on the visualization. Respond with valid JSON only."
        )
