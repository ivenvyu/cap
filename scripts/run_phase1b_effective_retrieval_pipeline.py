from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run_step(name: str, cmd: list[str]) -> dict:
    print("\n" + "=" * 80)
    print(name)
    print(" ".join(cmd))
    print("=" * 80)

    started_at = now()
    subprocess.run(cmd, check=True)

    return {
        "step": name,
        "cmd": cmd,
        "started_at": started_at,
        "finished_at": now(),
    }


def require(path: str) -> None:
    if not Path(path).exists():
        raise RuntimeError(f"필수 파일이 없음: {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/db/cap_reranker_ontology.db")
    ap.add_argument(
        "--cluster-label-csv",
        default="annotations/ontology/cluster_label_queue__coarse.reviewed_v1.csv",
    )
    ap.add_argument(
        "--knowledge",
        default="configs/domain_knowledge/sayuwon_entity_knowledge_v1.json",
    )
    ap.add_argument("--policy-id", default="seasonal_prompt_cue_policy_v1")
    ap.add_argument("--text-model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--skip-clip-cue-scoring", action="store_true")
    ap.add_argument(
        "--summary-out",
        default="audit/ontology/phase1b_effective_retrieval_pipeline_run_v1.summary.json",
    )
    args = ap.parse_args()

    require(args.cluster_label_csv)
    require(args.knowledge)
    require(f"configs/{args.policy_id}.yaml")

    py = sys.executable
    db = args.db
    steps = []

    steps.append(run_step("ontology DB 생성", [
        py, "scripts/build_ontology_db.py",
        "--db", db,
    ]))

    steps.append(run_step("ontology DB 기본 검증", [
        py, "scripts/validate_ontology_db.py",
        "--db", db,
    ]))

    steps.append(run_step("cluster label queue 생성", [
        py, "scripts/build_cluster_label_queue_from_db.py",
        "--db", db,
        "--cluster-level", "coarse",
    ]))

    steps.append(run_step("검토 완료 cluster tag ingest", [
        py, "scripts/ingest_cluster_tag_labels_to_db.py",
        "--db", db,
        "--label-csv", args.cluster_label_csv,
    ]))

    steps.append(run_step("cluster tag를 image tag로 전파", [
        py, "scripts/propagate_cluster_tags_to_images.py",
        "--db", db,
        "--cluster-level", "coarse",
    ]))

    steps.append(run_step("ontology tag assertion 검증", [
        py, "scripts/validate_ontology_tag_assertions.py",
        "--db", db,
        "--cluster-level", "coarse",
        "--min-cluster-tag-assertions", "1",
        "--min-image-tag-assertions", "1",
    ]))

    steps.append(run_step("사유원 식물 개화시기 prior ingest", [
        py, "scripts/ingest_sayuwon_plant_bloom_priors_to_db.py",
        "--db", db,
        "--knowledge", args.knowledge,
    ]))

    steps.append(run_step("계절 visual cue policy ingest", [
        py, "scripts/ingest_seasonal_prompt_cue_policy_to_db.py",
        "--db", db,
        "--policy", f"configs/{args.policy_id}.yaml",
    ]))

    if not args.skip_clip_cue_scoring:
        steps.append(run_step("CLIP visual cue score 계산", [
            py, "scripts/score_campaign_visual_cues_from_clip.py",
            "--db", db,
            "--policy-id", args.policy_id,
            "--text-model", args.text_model,
        ]))

    steps.append(run_step("식물 개화시기 campaign prior 계산", [
        py, "scripts/score_botanical_bloom_season_priors_from_db.py",
        "--db", db,
    ]))

    steps.append(run_step("꽃 계절 불일치 exclusion 생성", [
        py, "scripts/build_flower_season_exclusions_from_db.py",
        "--db", db,
    ]))

    steps.append(run_step("retrieval 후보에 flower-season exclusion 적용", [
        py, "scripts/apply_flower_season_exclusions_to_retrieval_candidates.py",
        "--db", db,
    ]))

    steps.append(run_step("effective retrieval view 생성", [
        py, "scripts/build_effective_retrieval_views_from_db.py",
        "--db", db,
    ]))

    steps.append(run_step("effective training view 생성", [
        py, "scripts/build_effective_training_views_from_db.py",
        "--db", db,
    ]))

    steps.append(run_step("ontology effective search index export", [
        py, "scripts/export_ontology_effective_search_index_from_db.py",
        "--db", db,
    ]))

    steps.append(run_step("effective search index query smoke test", [
        py, "scripts/query_smoke_ontology_effective_search_index.py",
        "--index", "data/search/ontology_effective_search_index_v1.jsonl",
    ]))

    steps.append(run_step("최종 ontology DB 검증", [
        py, "scripts/validate_ontology_db.py",
        "--db", db,
    ]))

    summary = {
        "event": "done",
        "pipeline": "phase1b_effective_retrieval_pipeline_v1",
        "db": db,
        "steps": steps,
        "outputs": {
            "search_index": "data/search/ontology_effective_search_index_v1.jsonl",
            "query_smoke_report": "audit/ontology/ontology_effective_search_index_query_smoke_v1.summary.json",
            "pipeline_summary": args.summary_out,
        },
        "해석": "Phase 1b effective retrieval 상태를 DB 기준으로 재현하는 실행 runner다.",
    }

    out = Path(args.summary_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
