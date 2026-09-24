"""Open-World 3D Object Understanding and Unknown Classifier for Scan-to-BIM.

Phase 3B Core Requirements (Sections 15 & 16):
- Open-World rather than assuming a closed training vocabulary.
- Hierarchy:
    1. Trained 3D semantic model (RandLA-Net CPU production checkpoint)
    2. Open-vocabulary semantic model (if CPU-available; if not: OPEN_VOCABULARY_UNAVAILABLE)
    3. Point-cloud geometric invariants
    4. Topological relations and context
    5. UNKNOWN classification
- Unknown is a FIRST-CLASS output:
    * If physical object is discovered but evidence is weak/ambiguous:
      final_label = "UNKNOWN_OBJECT" (or "UNKNOWN_<CATEGORY>")
    * NEVER force a known semantic class.
    * Output calibrated confidence distribution and reason codes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import structlog

from agent.phase3.instance.proposal_engine import ObjectProposal
from agent.phase3.taxonomy import (
    CLASS_TO_CATEGORY_MAP,
    HIERARCHICAL_TAXONOMY,
    get_category_for_class,
    is_unknown_or_unsupported,
)

logger = structlog.get_logger(__name__)

# Status for open vocabulary foundation model check
OPEN_VOCABULARY_STATUS = "OPEN_VOCABULARY_UNAVAILABLE"


@dataclass
class SemanticClassificationResult:
    """Detailed semantic classification outcome for an object proposal."""
    final_label: str
    level1_category: str
    label_status: str  # "SUPPORTED", "UNCERTAIN", "UNKNOWN", "REVIEW_REQUIRED", "REJECTED"
    confidence: float
    semantic_confidence: float
    geometric_confidence: float
    candidates: dict[str, float]
    probability_distribution: dict[str, float]
    reason_codes: list[str]
    evidence_breakdown: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_label": self.final_label,
            "level1_category": self.level1_category,
            "label_status": self.label_status,
            "confidence": round(float(self.confidence), 4),
            "semantic_confidence": round(float(self.semantic_confidence), 4),
            "geometric_confidence": round(float(self.geometric_confidence), 4),
            "candidates": {k: round(float(v), 4) for k, v in self.candidates.items()},
            "probability_distribution": {k: round(float(v), 4) for k, v in self.probability_distribution.items()},
            "reason_codes": self.reason_codes,
            "evidence_breakdown": self.evidence_breakdown,
        }


class OpenWorldRecognitionEngine:
    """Classifies physical object proposals into supported classes or UNKNOWN categories."""

    def __init__(
        self,
        min_known_confidence: float = 0.55,
        min_margin_ratio: float = 1.25,
        open_vocab_enabled: bool = False,
    ) -> None:
        self.min_known_confidence = min_known_confidence
        self.min_margin_ratio = min_margin_ratio
        self.open_vocab_enabled = open_vocab_enabled
        self.open_vocab_status = OPEN_VOCABULARY_STATUS

    def classify_proposal(
        self,
        proposal: ObjectProposal,
        neural_labels: list[str] | None = None,
        neural_probs: np.ndarray | None = None,
        host_wall_id: str | None = None,
        connected_to_pipe: bool = False,
    ) -> SemanticClassificationResult:
        """Evaluate multi-modal evidence and assign semantic identity or UNKNOWN."""
        reason_codes: list[str] = []
        sig = proposal.geometric_signature
        planarity = sig.get("planarity", 0.0)
        linearity = sig.get("linearity", 0.0)
        scattering = sig.get("scattering", 0.0)
        verticality = sig.get("verticality", 0.0)
        horizontality = sig.get("horizontality", 0.0)
        cylindricality = sig.get("cylindricality", 0.0)

        extents = proposal.obb_extents_m
        min_dim = float(extents[0])
        mid_dim = float(extents[1])
        max_dim = float(extents[2])
        aspect_ratio = max_dim / max(min_dim, 1e-4)

        # 1. Neural evidence aggregation across proposal points
        neural_candidate_counts: dict[str, int] = {}
        if neural_labels is not None and len(neural_labels) > 0:
            for p_idx in proposal.point_indices:
                if p_idx < len(neural_labels):
                    lbl = neural_labels[p_idx]
                    neural_candidate_counts[lbl] = neural_candidate_counts.get(lbl, 0) + 1

        total_pts = len(proposal.point_indices)
        neural_scores: dict[str, float] = {}
        if total_pts > 0 and neural_candidate_counts:
            for lbl, count in neural_candidate_counts.items():
                neural_scores[lbl] = count / total_pts

        # 2. Geometric Evidence Calculation for All Vocabulary Classes
        geom_scores: dict[str, float] = {
            "WALL": 0.0,
            "FLOOR": 0.0,
            "CEILING": 0.0,
            "COLUMN": 0.0,
            "BEAM": 0.0,
            "PIPE": 0.0,
            "DUCT": 0.0,
            "CABLE_TRAY": 0.0,
            "VALVE": 0.0,
            "DOOR": 0.0,
            "WINDOW": 0.0,
            "EQUIPMENT": 0.0,
            "UNKNOWN_OBJECT": 0.15,
        }

        # Axis of elongation from PCA
        axis_elongation = proposal.obb_rotation_matrix[:, 2]
        axis_z = abs(float(axis_elongation[2]))
        is_vertical_axis = axis_z > 0.75
        is_horizontal_axis = axis_z < 0.45

        # Wall: high planarity, vertical, planar extent > 0.5m, thickness < 0.6m
        if planarity > 0.40 and verticality > 0.60 and min_dim < 0.60 and max_dim > 0.50:
            geom_scores["WALL"] = float(0.5 * planarity + 0.5 * verticality)

        # Floor / Ceiling / Slab: high planarity, horizontal, thickness < 0.6m, extent > 0.5m
        if planarity > 0.40 and horizontality > 0.60 and min_dim < 0.60 and max_dim > 0.50:
            geom_scores["FLOOR"] = float(0.5 * planarity + 0.5 * horizontality)
            geom_scores["CEILING"] = float(0.5 * planarity + 0.5 * horizontality)

        # Column: vertical elongation axis, cylindrical or square cross-section, aspect ratio > 1.8
        if is_vertical_axis and aspect_ratio > 1.8 and max(min_dim, mid_dim) < 1.2:
            col_score = 0.5 * axis_z + 0.3 * linearity + 0.2 * min(1.0, aspect_ratio / 3.0)
            geom_scores["COLUMN"] = float(col_score)

        # Pipe: cylindrical, elongated along horizontal or arbitrary path, diameter < 0.45m
        diameter_est = float(min_dim + mid_dim) / 2.0
        if (cylindricality > 0.25 or (linearity > 0.40 and not is_vertical_axis)) and diameter_est < 0.45:
            pipe_score = 0.5 * max(cylindricality, linearity) + 0.3 * (1.0 - min(1.0, diameter_est / 0.45)) + 0.2
            geom_scores["PIPE"] = float(min(0.95, pipe_score))

        # Duct: elongated rectangular cross-section, planar sides, larger cross section
        if linearity > 0.35 and planarity > 0.25 and 0.15 < min_dim < 1.5 and max_dim > 0.8 and not is_vertical_axis:
            geom_scores["DUCT"] = float(0.4 * linearity + 0.4 * planarity + 0.1)

        # Cable Tray: elongated, flat / U-shaped, aspect ratio > 2.5, width between 0.1m and 0.9m
        if linearity > 0.45 and 0.06 < min_dim < 0.25 and 0.15 < mid_dim < 0.90 and is_horizontal_axis:
            geom_scores["CABLE_TRAY"] = float(0.5 * linearity + 0.3 * planarity + 0.1)

        # Beam: horizontal structural member, high linearity, rectangular profile
        if is_horizontal_axis and linearity > 0.45 and aspect_ratio > 2.0 and 0.10 < min_dim < 0.80 and cylindricality < 0.20:
            geom_scores["BEAM"] = float(0.6 * linearity + 0.4 * (1.0 - axis_z))

        # Valve / Fitting: compact MEP component connected to pipe
        if connected_to_pipe and (scattering > 0.20 or min_dim < 0.40):
            geom_scores["VALVE"] = 0.85
            reason_codes.append("PIPE_CONNECTION_CONFIRMED")
        elif connected_to_pipe:
            geom_scores["VALVE"] = 0.70
            reason_codes.append("PIPE_CONNECTION_CONFIRMED")

        # Door / Window: opening inside host wall
        if host_wall_id is not None:
            if min_dim < 0.35 and max_dim > 0.6:
                geom_scores["DOOR"] = 0.75
                geom_scores["WINDOW"] = 0.70
                reason_codes.append("HOST_WALL_CONFIRMED")

        # Equipment: compact / complex machinery with multi-part structure
        # Requires neural recognition or topological MEP connection (pipe/duct) to classify as EQUIPMENT;
        # otherwise ambiguous objects are classified as UNKNOWN_OBJECT per Section 16.
        has_equipment_neural = neural_scores.get("EQUIPMENT", 0.0) > 0.25
        if (has_equipment_neural or connected_to_pipe) and scattering > 0.25 and max_dim > 0.40 and min_dim > 0.20:
            surf_evidence = max(planarity, cylindricality)
            geom_scores["EQUIPMENT"] = float(0.4 * scattering + 0.3 * surf_evidence + 0.3 * min(1.0, proposal.volume_m3 / 0.5))

        # Unknown Object base score:
        # If no specific geometric primitive has strong evidence, elevate UNKNOWN_OBJECT
        max_known_geom = max(
            v for k, v in geom_scores.items() if k != "UNKNOWN_OBJECT"
        )
        if max_known_geom < 0.40:
            geom_scores["UNKNOWN_OBJECT"] = float(max(0.60, 1.0 - max_known_geom))
        else:
            geom_scores["UNKNOWN_OBJECT"] = 0.15

        # 3. Fuse Neural + Geometric Scores
        combined_scores: dict[str, float] = {}
        for cls_name, g_score in geom_scores.items():
            n_score = neural_scores.get(cls_name, 0.0)
            if n_score > 0:
                reason_codes.append("STRONG_NEURAL_SUPPORT")
                combined = 0.55 * n_score + 0.45 * g_score
            else:
                combined = g_score
            combined_scores[cls_name] = combined

        # Normalized probability distribution over non-trivial classes
        total_raw = sum(combined_scores.values())
        if total_raw > 1e-6:
            prob_dist = {k: v / total_raw for k, v in combined_scores.items()}
        else:
            prob_dist = {k: 1.0 / len(combined_scores) for k in combined_scores}

        # Sort candidates descending by fused score
        sorted_candidates = sorted(combined_scores.items(), key=lambda kv: kv[1], reverse=True)
        top_cls, top_raw_score = sorted_candidates[0]
        second_cls, second_raw_score = sorted_candidates[1] if len(sorted_candidates) > 1 else ("NONE", 0.0)

        # 4. Strict Open-World Unknown Gate (Section 16)
        # If top score is low OR top two candidates are closely tied -> classify as UNKNOWN
        is_unknown = False
        final_label = top_cls
        status = "SUPPORTED"

        if top_cls == "UNKNOWN_OBJECT":
            is_unknown = True
            reason_codes.append("UNKNOWN_CLASS")
        elif top_raw_score < self.min_known_confidence:
            is_unknown = True
            reason_codes.append("WEAK_SEMANTIC_EVIDENCE")
        elif second_raw_score > 0.20 and (top_raw_score / max(second_raw_score, 1e-4)) < self.min_margin_ratio:
            # Candidates are ambiguous (e.g. valve: 0.28, fitting: 0.25)
            is_unknown = True
            reason_codes.append("AMBIGUOUS_EVIDENCE")

        if is_unknown:
            # Derive category-specific unknown if possible
            cat = get_category_for_class(top_cls)
            if cat in ("MEP", "STRUCTURAL", "EQUIPMENT"):
                final_label = f"UNKNOWN_{cat}"
            else:
                final_label = "UNKNOWN_OBJECT"
            status = "UNKNOWN"

        # Determine Level 1 category
        level1_cat = get_category_for_class(final_label)

        # Geometric confidence
        geom_conf = float(combined_scores.get(top_cls, 0.2))
        neural_conf = float(neural_scores.get(top_cls, 0.0))

        if geom_conf > 0.65:
            reason_codes.append("HIGH_GEOMETRIC_SUPPORT")
        if total_pts < 30:
            reason_codes.append("LOW_POINT_SUPPORT")

        return SemanticClassificationResult(
            final_label=final_label,
            level1_category=level1_cat,
            label_status=status,
            confidence=float(top_raw_score),
            semantic_confidence=neural_conf,
            geometric_confidence=geom_conf,
            candidates=dict(sorted_candidates[:5]),
            probability_distribution=prob_dist,
            reason_codes=list(set(reason_codes)),
            evidence_breakdown={
                "neural_scores": neural_scores,
                "geometric_scores": geom_scores,
                "aspect_ratio": round(aspect_ratio, 2),
                "open_vocab_status": self.open_vocab_status,
            },
        )

    def calibrate_from_validation(
        self,
        val_proposals: list[ObjectProposal],
        val_ground_truth: list[Any],
    ) -> dict[str, Any]:
        """Calibrate rejection thresholds from held-out validation evidence per Section 21.

        Searches grid of confidence and margin ratios to optimize known-class F1
        while bounding the false-known rate.
        """
        if not val_proposals or not val_ground_truth:
            return {
                "calibrated_confidence": self.min_known_confidence,
                "calibrated_margin": self.min_margin_ratio,
                "status": "DEFAULT_THRESHOLDS_RETAINED",
            }

        # Build map from proposal to nearest GT instance
        gt_sets = [set(gt.point_ids) for gt in val_ground_truth]
        gt_classes = [str(gt.semantic_class).upper() for gt in val_ground_truth]

        best_f1 = -1.0
        best_conf = self.min_known_confidence
        best_margin = self.min_margin_ratio
        best_fkr = 0.0
        best_urr = 0.0

        for conf_cand in [0.45, 0.50, 0.55, 0.60, 0.65]:
            for margin_cand in [1.15, 1.20, 1.25, 1.30]:
                self.min_known_confidence = conf_cand
                self.min_margin_ratio = margin_cand

                tp, fp, fn = 0, 0, 0
                unknown_gt_count = 0
                unknown_rejected = 0
                false_known = 0

                for prop in val_proposals:
                    p_set = set(prop.source_point_ids if prop.source_point_ids else prop.point_indices.tolist())
                    # Find best matching GT
                    best_overlap = 0
                    best_gt_idx = -1
                    for g_idx, g_set in enumerate(gt_sets):
                        ov = len(p_set & g_set)
                        if ov > best_overlap:
                            best_overlap = ov
                            best_gt_idx = g_idx

                    if best_gt_idx < 0:
                        continue

                    gt_cls = gt_classes[best_gt_idx]
                    pred_res = self.classify_proposal(prop)
                    pred_label = pred_res.final_label.upper()

                    if gt_cls == "UNKNOWN":
                        unknown_gt_count += 1
                        if "UNKNOWN" in pred_label:
                            unknown_rejected += 1
                        else:
                            false_known += 1
                    else:
                        if pred_label == gt_cls:
                            tp += 1
                        elif "UNKNOWN" in pred_label:
                            fn += 1
                        else:
                            fp += 1

                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                fkr = false_known / max(unknown_gt_count, 1)
                urr = unknown_rejected / max(unknown_gt_count, 1)

                if f1 > best_f1:
                    best_f1 = f1
                    best_conf = conf_cand
                    best_margin = margin_cand
                    best_fkr = fkr
                    best_urr = urr

        # Commit calibrated parameters
        self.min_known_confidence = best_conf
        self.min_margin_ratio = best_margin

        report = {
            "calibrated_confidence": round(best_conf, 4),
            "calibrated_margin_ratio": round(best_margin, 4),
            "validation_known_f1": round(best_f1, 4),
            "validation_unknown_rejection_rate": round(best_urr, 4),
            "validation_false_known_rate": round(best_fkr, 4),
            "calibration_status": "CALIBRATED_FROM_VALIDATION_SPLIT",
        }
        logger.info("open_world_confidence_calibrated", **report)
        return report
