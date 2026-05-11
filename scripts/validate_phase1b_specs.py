from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"missing file: {path}")
    if path.stat().st_size == 0:
        raise RuntimeError(f"empty file: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        raise RuntimeError(f"yaml parsed as None: {path}")
    if not isinstance(data, dict):
        raise RuntimeError(f"yaml root must be object: {path}")
    return data


def require_keys(obj: dict[str, Any], keys: list[str], where: str) -> None:
    for key in keys:
        if key not in obj:
            raise RuntimeError(f"{where}: missing required key: {key}")


def validate_diagnostic_claim_support() -> None:
    path = ROOT / "configs/diagnostic_claim_support_v1.yaml"
    data = load_yaml(path)

    require_keys(
        data,
        ["metadata", "name", "scope", "claim_schema",
         "threshold_status_semantics", "entry_validation_policy", "claims"],
        "diagnostic_claim_support_v1",
    )

    if data["name"] != "diagnostic_claim_support_v1":
        raise RuntimeError("diagnostic_claim_support_v1: bad name")

    metadata = data["metadata"]
    require_keys(metadata, ["spec_version", "phase_introduced", "status"], "metadata")

    if metadata["spec_version"] != "v2.2.1":
        raise RuntimeError("spec_version must be v2.2.1")
    if metadata["phase_introduced"] != "phase_1b":
        raise RuntimeError("phase_introduced must be phase_1b")

    scope = data["scope"]
    require_keys(scope, ["support_level", "excluded_level", "explanation_target",
                          "score_status", "threshold_status"], "scope")
    if scope["support_level"] != "claim_level":
        raise RuntimeError("support_level must be claim_level")
    if scope["excluded_level"] != "candidate_level":
        raise RuntimeError("excluded_level must be candidate_level")
    if scope["score_status"] != "diagnostic_only":
        raise RuntimeError("score_status must be diagnostic_only")
    if scope["threshold_status"] != "diagnostic_trigger_only_not_final_threshold":
        raise RuntimeError("scope.threshold_status must be diagnostic_trigger_only_not_final_threshold")

    # threshold_status_semantics
    threshold_semantics = data["threshold_status_semantics"]
    if not isinstance(threshold_semantics, dict) or not threshold_semantics:
        raise RuntimeError("threshold_status_semantics must be non-empty object")
    known_threshold_statuses = set(threshold_semantics.keys())

    for status_name, spec in threshold_semantics.items():
        if not isinstance(spec, dict):
            raise RuntimeError(f"threshold_status_semantics.{status_name}: must be object")
        require_keys(spec, ["meaning", "requires_next_action"],
                     f"threshold_status_semantics.{status_name}")
        if not isinstance(spec["requires_next_action"], bool):
            raise RuntimeError(
                f"threshold_status_semantics.{status_name}.requires_next_action must be bool"
            )

    # entry_validation_policy
    validation_policy = data["entry_validation_policy"]
    require_keys(validation_policy,
                 ["require_next_action_when_threshold_status_in",
                  "allow_record_only_next_action_when_threshold_status_in"],
                 "entry_validation_policy")

    requires_next_action_statuses = set(
        validation_policy["require_next_action_when_threshold_status_in"]
    )
    record_only_statuses = set(
        validation_policy["allow_record_only_next_action_when_threshold_status_in"]
    )

    unknown_required = requires_next_action_statuses - known_threshold_statuses
    if unknown_required:
        raise RuntimeError(
            f"entry_validation_policy has unknown require-next-action statuses: {sorted(unknown_required)}"
        )
    unknown_record_only = record_only_statuses - known_threshold_statuses
    if unknown_record_only:
        raise RuntimeError(
            f"entry_validation_policy has unknown record-only statuses: {sorted(unknown_record_only)}"
        )

    # Cross-check: semantics and policy must agree.
    semantic_required = {
        status for status, spec in threshold_semantics.items()
        if spec["requires_next_action"] is True
    }
    semantic_record_only = {
        status for status, spec in threshold_semantics.items()
        if spec["requires_next_action"] is False
    }

    if requires_next_action_statuses & record_only_statuses:
        raise RuntimeError(
            "entry_validation_policy require-next-action and record-only sets overlap: "
            f"{sorted(requires_next_action_statuses & record_only_statuses)}"
        )

    if semantic_required != requires_next_action_statuses:
        raise RuntimeError(
            f"threshold_status_semantics.requires_next_action mismatch with "
            f"entry_validation_policy: semantic={sorted(semantic_required)}, "
            f"policy={sorted(requires_next_action_statuses)}"
        )

    if semantic_record_only != record_only_statuses:
        raise RuntimeError(
            f"threshold_status_semantics record-only mismatch with "
            f"entry_validation_policy: semantic={sorted(semantic_record_only)}, "
            f"policy={sorted(record_only_statuses)}"
        )

    for status in semantic_record_only:
        spec = threshold_semantics[status]
        if "default_next_action" not in spec:
            raise RuntimeError(
                f"threshold_status_semantics.{status}: record-only status requires default_next_action"
            )
        default_next_action = spec["default_next_action"]
        if not isinstance(default_next_action, str) or not default_next_action.strip():
            raise RuntimeError(
                f"threshold_status_semantics.{status}.default_next_action must be non-empty string"
            )

    # claim_schema
    claim_schema = data["claim_schema"]
    require_keys(claim_schema, ["required_fields", "allowed_status", "allowed_claim_types"],
                 "claim_schema")
    required_claim_fields = set(claim_schema["required_fields"])
    allowed_status = set(claim_schema["allowed_status"])
    allowed_claim_types = set(claim_schema["allowed_claim_types"])

    # claims
    claims = data["claims"]
    if not isinstance(claims, dict) or not claims:
        raise RuntimeError("claims must be non-empty object")

    for claim_id, claim in claims.items():
        if not isinstance(claim, dict):
            raise RuntimeError(f"claims.{claim_id}: must be object")

        missing = required_claim_fields - set(claim.keys())
        if missing:
            raise RuntimeError(f"claims.{claim_id}: missing required fields: {sorted(missing)}")

        # claim_id field must match dict key
        if claim.get("claim_id") != claim_id:
            raise RuntimeError(
                f"claims.{claim_id}: claim_id field must match key, "
                f"got {claim.get('claim_id')!r}"
            )

        if claim["status"] not in allowed_status:
            raise RuntimeError(
                f"claims.{claim_id}: invalid status {claim['status']!r}; "
                f"allowed={sorted(allowed_status)}"
            )
        if claim["claim_type"] not in allowed_claim_types:
            raise RuntimeError(
                f"claims.{claim_id}: invalid claim_type {claim['claim_type']!r}; "
                f"allowed={sorted(allowed_claim_types)}"
            )

        # would_strengthen / would_weaken entries
        for direction in ["would_strengthen", "would_weaken"]:
            entries = claim.get(direction, [])
            if not isinstance(entries, list):
                raise RuntimeError(f"claims.{claim_id}.{direction}: must be list")

            for idx, entry in enumerate(entries):
                where = f"claims.{claim_id}.{direction}[{idx}]"
                if not isinstance(entry, dict):
                    raise RuntimeError(f"{where}: must be object")

                require_keys(entry, ["observation", "measurement", "threshold_status"], where)

                threshold_status = entry["threshold_status"]
                if threshold_status not in known_threshold_statuses:
                    raise RuntimeError(
                        f"{where}: unknown threshold_status={threshold_status!r}; "
                        f"known={sorted(known_threshold_statuses)}"
                    )

                if threshold_status in requires_next_action_statuses:
                    if "next_action" not in entry:
                        raise RuntimeError(
                            f"{where}: threshold_status={threshold_status!r} requires next_action"
                        )
                    next_action = entry["next_action"]
                    if not isinstance(next_action, str) or not next_action.strip():
                        raise RuntimeError(f"{where}: next_action must be non-empty string")

                if threshold_status in record_only_statuses:
                    if "next_action" in entry:
                        next_action = entry["next_action"]
                        if not isinstance(next_action, str) or not next_action.strip():
                            raise RuntimeError(
                                f"{where}: optional next_action must be non-empty string"
                            )

        # forbidden_interpretation
        forbidden = claim.get("forbidden_interpretation", [])
        if not isinstance(forbidden, list) or not forbidden:
            raise RuntimeError(
                f"claims.{claim_id}: forbidden_interpretation must be non-empty list"
            )

    print(f"diagnostic_claim_support_v1 ok — {len(claims)} claims validated")



def validate_layout_treatment_preview_policy() -> None:
    path = ROOT / "configs/layout_treatment_preview_policy_v1.yaml"
    data = load_yaml(path)

    require_keys(
        data,
        [
            "metadata",
            "name",
            "scope",
            "core_decision",
            "separation_rules",
            "contrast_reference",
            "treatment_options",
            "preview_variants",
            "layout_treatment_review_event",
            "non_claims",
        ],
        "layout_treatment_preview_policy_v1",
    )

    if data["name"] != "layout_treatment_preview_policy_v1":
        raise RuntimeError("layout_treatment_preview_policy_v1: bad name")

    metadata = data["metadata"]
    require_keys(metadata, ["spec_version", "phase_introduced", "status"], "layout_treatment.metadata")
    if metadata["spec_version"] != "v2.2.1":
        raise RuntimeError("layout_treatment_preview_policy_v1: spec_version must be v2.2.1")
    if metadata["phase_introduced"] != "phase_1b":
        raise RuntimeError("layout_treatment_preview_policy_v1: phase_introduced must be phase_1b")

    scope = data["scope"]
    require_keys(
        scope,
        [
            "renderer_role",
            "preview_object",
            "score_status",
            "candidate_support_explanation_status",
            "final_design_renderer",
            "final_poster_quality_claim",
        ],
        "layout_treatment.scope",
    )

    if scope["score_status"] != "diagnostic_only":
        raise RuntimeError("layout_treatment_preview_policy_v1: score_status must be diagnostic_only")
    if scope["candidate_support_explanation_status"] != "deferred_from_phase_1b":
        raise RuntimeError(
            "layout_treatment_preview_policy_v1: candidate_support_explanation_status must be deferred_from_phase_1b"
        )
    if scope["final_design_renderer"] is not False:
        raise RuntimeError("layout_treatment_preview_policy_v1: final_design_renderer must be false")
    if scope["final_poster_quality_claim"] is not False:
        raise RuntimeError("layout_treatment_preview_policy_v1: final_poster_quality_claim must be false")

    contrast = data["contrast_reference"]
    require_keys(
        contrast,
        [
            "source_type",
            "standard_name",
            "scope",
            "normal_text_min_contrast_ratio",
            "large_text_min_contrast_ratio",
        ],
        "layout_treatment.contrast_reference",
    )

    if contrast["source_type"] != "external_standard_explicitly_cited_and_scoped":
        raise RuntimeError(
            "layout_treatment_preview_policy_v1: contrast_reference.source_type must be external_standard_explicitly_cited_and_scoped"
        )
    if contrast["standard_name"] != "WCAG_2_2_SC_1_4_3_Contrast_Minimum":
        raise RuntimeError("layout_treatment_preview_policy_v1: unsupported contrast standard")
    if float(contrast["normal_text_min_contrast_ratio"]) != 4.5:
        raise RuntimeError("layout_treatment_preview_policy_v1: normal text contrast ratio must be 4.5")
    if float(contrast["large_text_min_contrast_ratio"]) != 3.0:
        raise RuntimeError("layout_treatment_preview_policy_v1: large text contrast ratio must be 3.0")

    required_treatments = {
        "use_as_is",
        "needs_dark_scrim",
        "needs_light_scrim",
        "needs_gradient_scrim",
        "needs_split_panel",
        "needs_text_color_swap",
        "needs_manual_design",
        "unusable_even_with_treatment",
    }

    treatment_options = data["treatment_options"]
    missing_treatments = required_treatments - set(treatment_options.keys())
    if missing_treatments:
        raise RuntimeError(
            f"layout_treatment_preview_policy_v1: missing treatment_options {sorted(missing_treatments)}"
        )

    event = data["layout_treatment_review_event"]
    require_keys(
        event,
        ["required_fields", "allowed_layout_issue_tags", "allowed_layout_treatment_decisions"],
        "layout_treatment.layout_treatment_review_event",
    )

    event_decisions = set(event["allowed_layout_treatment_decisions"])
    if event_decisions != required_treatments:
        raise RuntimeError(
            "layout_treatment_preview_policy_v1: allowed_layout_treatment_decisions must match treatment_options"
        )

    separation_rules = set(data["separation_rules"])
    required_rules = {
        "text_region_conflict_is_not_immediate_reject",
        "low_contrast_is_not_immediate_reject",
        "too_busy_background_is_not_immediate_reject",
        "layout_issue_tags_indicate_treatment_need",
        "campaign_unsuitability_and_layout_treatment_need_must_be_recorded_separately",
    }
    missing_rules = required_rules - separation_rules
    if missing_rules:
        raise RuntimeError(
            f"layout_treatment_preview_policy_v1: missing separation_rules {sorted(missing_rules)}"
        )

    print("layout_treatment_preview_policy_v1 ok")

def main() -> None:
    print("=== Phase 1b spec validation ===\n")
    validate_diagnostic_claim_support()
    validate_layout_treatment_preview_policy()
    print("\nALL PHASE 1B SPEC CHECKS PASSED")


if __name__ == "__main__":
    main()
