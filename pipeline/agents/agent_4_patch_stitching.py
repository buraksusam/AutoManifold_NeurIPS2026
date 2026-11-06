"""
Worker Agent 4 — Patch Stitching Designer Agent

DR_Template Section 1.3: Design Local Patch Stitching Algorithm.
Designs methods to assemble locally-embedded patches into a globally
consistent embedding (position, rotation, reflection, scale).
Prioritises reuse of existing mathematical solutions.
"""

from ..base_agent import BaseAgent

PROMPT_TEMPLATE = """You are the Patch Stitching Designer Agent. Your task is to design an algorithm that stitches locally-embedded patches into a coherent global embedding — following Section 1.3 of the DR_Template.

CONTEXT (from DR_Template Section 1.3):
"Divide data points into local patches, assuming each local patch of data points already have a decent internal neighbor order. Design patch stitching algorithms to find the relative position, rotation, reflection, and scale of the local patches so the global neighbor order is as consistent as possible. Consider both situations where patches have and do not have overlapping points."

CRITICAL INSTRUCTION (from DR_Template):
"Find existing mathematical problem closest to this setup and re-use existing solutions. Do NOT design new algorithms if not having to."

NOTE: The DR_Template authors tested the prompt "There are three local patches of data points. Within each patch, the points have fairly correct neighbour order. How to put these three local patches in the space so that all the points have fairly correct neighbour order?" on ChatGPT and found the suggested design reasonable.

INPUT:
- Neighborhood analysis: {neighborhood_summary}
- Number of patches: {n_patches}
- Recommended patch size: {patch_size}
- Overlap information: {overlap_info}
- Per-patch embedding quality: {patch_quality}

TASKS:
1. Identify the closest existing mathematical formulations. Consider:
   - Procrustes analysis (rotation, reflection, scaling between point sets)
   - Generalized Procrustes analysis (> 2 patches simultaneously)
   - Graph matching / graph alignment
   - Manifold alignment
   - Spectral methods for multi-patch alignment
2. For the OVERLAPPING case: design stitching using shared points as anchors.
3. For the NON-OVERLAPPING case: design using inter-patch HD distance or neighborhood info.
4. Handle all four transformation components: translation, rotation, reflection, scale.

REQUIRED OUTPUT (JSON):
{{
  "closest_existing_formulations": [
    {{
      "problem_name": "string",
      "similarity_to_our_problem": "string",
      "reference": "string",
      "applicable_components": ["translation", "rotation", "reflection", "scaling"],
      "limitations": "string"
    }}
  ],
  "stitching_algorithm": {{
    "name": "string",
    "type": "existing_solution | adaptation | novel_design",
    "mathematical_basis": "string",
    "handles_overlap": true,
    "handles_no_overlap": true
  }},
  "algorithm_steps": {{
    "with_overlap": [
      {{
        "step": int,
        "description": "string",
        "mathematical_detail": "string",
        "pseudocode": "string"
      }}
    ],
    "without_overlap": [
      {{
        "step": int,
        "description": "string",
        "mathematical_detail": "string",
        "pseudocode": "string"
      }}
    ]
  }},
  "transformation_estimation": {{
    "rotation_method": "string",
    "reflection_detection": "string",
    "scaling_method": "string",
    "translation_method": "string"
  }},
  "global_consistency_optimization": {{
    "objective": "string",
    "method": "string",
    "convergence_criterion": "string"
  }},
  "complexity_analysis": {{
    "time_complexity": "string",
    "space_complexity": "string",
    "scalability_notes": "string"
  }}
}}"""


class PatchStitchingAgent(BaseAgent):
    AGENT_NAME = "PatchStitching"
    AGENT_ID = 4

    def build_prompt(self, context: dict) -> str:
        a3 = context.get("agent3_output", {})
        stitching = a3.get("stitching_implications", {})
        return PROMPT_TEMPLATE.format(
            neighborhood_summary=a3.get("maximal_neighborhoods", "N/A"),
            n_patches=stitching.get("recommended_patch_size", "TBD"),
            patch_size=stitching.get("recommended_patch_size", 100),
            overlap_info=stitching.get("expected_overlap_needed", "TBD"),
            patch_quality=context.get("patch_quality", "N/A"),
        )

    def system_message(self) -> str:
        return (
            "You are a mathematical algorithm designer specialising in manifold alignment "
            "and point-cloud registration. You prioritise reusing well-known formulations "
            "(Procrustes, spectral alignment) over inventing new methods. Respond with valid JSON only."
        )
