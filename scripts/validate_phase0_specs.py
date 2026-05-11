from pathlib import Path
import json
import yaml


REQUIRED_FILES = [
    "configs/issue_tags_v1.yaml",
    "configs/review_queue_policy_v2_2.yaml",
    "configs/calibration_policy_v1.yaml",
    "configs/prompt_template_bank_v1.yaml",
    "configs/prompt_template_bank_governance_v1.yaml",
    "configs/critic_preview_renderer_v1.yaml",
    "configs/pair_feature_snapshot_storage_v1.yaml",
    "configs/phase_1a_exit_criteria.yaml",
    "configs/training_snapshot_aggregation_v1.yaml",
    "schemas/review_event.schema.json",
    "schemas/pair_feature_snapshot.schema.json",
    "schemas/training_snapshot.schema.json",
    "docs/design_history/v2_2_1_freeze.md",
]


def load_yaml(path: str):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if data is None:
        raise RuntimeError(f"YAML parsed as None: {path}")
    return data


def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    print("=== file existence / non-empty check ===")
    for f in REQUIRED_FILES:
        p = Path(f)
        if not p.exists():
            raise FileNotFoundError(f)
        size = p.stat().st_size
        if size == 0:
            raise RuntimeError(f"empty file: {f}")
        print(f"{f}: {size} bytes")

    print("\n=== parse check ===")
    yaml_files = [f for f in REQUIRED_FILES if f.endswith(".yaml")]
    json_files = [f for f in REQUIRED_FILES if f.endswith(".json")]

    yamls = {}
    for f in yaml_files:
        yamls[f] = load_yaml(f)
        print("YAML ok:", f)

    jsons = {}
    for f in json_files:
        jsons[f] = load_json(f)
        print("JSON ok:", f)

    print("\n=== metadata check ===")
    for f, data in yamls.items():
        meta = data.get("metadata")
        if not meta:
            raise RuntimeError(f"missing metadata: {f}")
        if str(meta.get("spec_version")) != "v2.2.1":
            raise RuntimeError(f"bad spec_version in {f}: {meta.get('spec_version')}")
        print("metadata ok:", f)

    for f, data in jsons.items():
        meta = data.get("x_metadata")
        if not meta:
            raise RuntimeError(f"missing x_metadata: {f}")
        if meta.get("spec_version") != "v2.2.1":
            raise RuntimeError(f"bad spec_version in {f}: {meta.get('spec_version')}")
        print("x_metadata ok:", f)

    print("\n=== issue tag consistency check ===")
    issue_cfg = yamls["configs/issue_tags_v1.yaml"]

    issue_tags = []
    for _, obj in issue_cfg["categories"].items():
        for tag in obj["tags"]:
            issue_tags.append(tag["id"])

    issue_tags = sorted(set(issue_tags))

    review_event_tags = sorted(
        jsons["schemas/review_event.schema.json"]
        ["properties"]["decision"]["properties"]["issue_tags"]["items"]["enum"]
    )

    training_tags = sorted(
        jsons["schemas/training_snapshot.schema.json"]
        ["properties"]["issue_tags"]["items"]["enum"]
    )

    if issue_tags != review_event_tags:
        raise RuntimeError("issue_tags_v1 != review_event schema issue tags")

    if issue_tags != training_tags:
        raise RuntimeError("issue_tags_v1 != training_snapshot schema issue tags")

    print("issue tag count:", len(issue_tags))
    print(issue_tags)

    print("\n=== phase_1a required files check ===")
    phase = yamls["configs/phase_1a_exit_criteria.yaml"]
    for f in phase["entry_requirements"]["required_files"]:
        if not Path(f).exists():
            raise FileNotFoundError(f"phase_1a required file missing: {f}")
        print("phase required ok:", f)


    print("\n=== bucket reference consistency check ===")
    review_policy = yamls["configs/review_queue_policy_v2_2.yaml"]
    bucket_defs = set(review_policy["bucket_definitions"].keys())
    source_bucket_enum = set(
        jsons["schemas/review_event.schema.json"]
        ["properties"]["review_context"]["properties"]["source_bucket"]["enum"]
    )

    missing_bucket_enums = sorted(bucket_defs - source_bucket_enum)
    if missing_bucket_enums:
        raise RuntimeError(f"bucket_definitions missing from ReviewEvent.source_bucket enum: {missing_bucket_enums}")

    meta_allocation_keys = {"model_disagreement"}
    for stage_name, stage in review_policy["stage_policy"].items():
        for bucket_name in stage["allocation"].keys():
            if bucket_name not in bucket_defs and bucket_name not in meta_allocation_keys:
                raise RuntimeError(f"stage_policy.{stage_name}.allocation references undefined bucket/meta-bucket: {bucket_name}")

    for availability_key in ["if_critic_available", "if_critic_unavailable"]:
        for bucket_name in review_policy["model_disagreement_policy"][availability_key]["priority"]:
            if bucket_name not in bucket_defs:
                raise RuntimeError(f"model_disagreement_policy.{availability_key} references undefined bucket: {bucket_name}")

    print("bucket definitions:", len(bucket_defs))
    print("bucket/source_bucket consistency ok")

    print("\n=== critic trainable tag consistency check ===")
    critic_trainable_tags = []
    for _, obj in issue_cfg["categories"].items():
        if obj.get("critic_trainable") is True:
            for tag in obj["tags"]:
                critic_trainable_tags.append(tag["id"])
    critic_trainable_tags = sorted(set(critic_trainable_tags))

    critic_preview = yamls["configs/critic_preview_renderer_v1.yaml"]
    preview_targets = sorted(set(critic_preview["critic_label_targets"]["trainable_issue_tags"]))

    aggregation = yamls["configs/training_snapshot_aggregation_v1.yaml"]
    aggregation_targets = sorted(set(aggregation["critic_label_mapping"]["positive_issue_tags"]))

    usage_targets = sorted(set(issue_cfg["training_usage"]["critic_supervision"]["positive_targets"]))

    if critic_trainable_tags != preview_targets:
        raise RuntimeError("critic_trainable tags != critic_preview_renderer targets")
    if critic_trainable_tags != aggregation_targets:
        raise RuntimeError("critic_trainable tags != training_snapshot_aggregation critic positives")
    if critic_trainable_tags != usage_targets:
        raise RuntimeError("critic_trainable tags != issue_tags training_usage critic targets")

    print("critic trainable tags:", critic_trainable_tags)

    print("\n=== decision label consistency check ===")
    review_labels = sorted(
        jsons["schemas/review_event.schema.json"]
        ["properties"]["decision"]["properties"]["label"]["enum"]
    )
    training_labels = sorted(
        jsons["schemas/training_snapshot.schema.json"]
        ["properties"]["decision_label"]["enum"]
    )
    if review_labels != training_labels:
        raise RuntimeError("ReviewEvent decision labels != TrainingSnapshot decision labels")
    print("decision labels:", review_labels)

    print("\n=== feature_snapshot_id pattern consistency check ===")
    patterns = {
        "pair_feature_snapshot": jsons["schemas/pair_feature_snapshot.schema.json"]["properties"]["feature_snapshot_id"]["pattern"],
        "review_event": jsons["schemas/review_event.schema.json"]["properties"]["feature_snapshot_id"]["pattern"],
        "training_snapshot": jsons["schemas/training_snapshot.schema.json"]["properties"]["feature_snapshot_id"]["pattern"],
    }
    if len(set(patterns.values())) != 1:
        raise RuntimeError(f"feature_snapshot_id patterns differ: {patterns}")
    print("feature_snapshot_id pattern ok:", next(iter(patterns.values())))


    print("\nALL PHASE 0 SPEC CHECKS PASSED")


if __name__ == "__main__":
    main()
