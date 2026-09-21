"""
scripts/ingest_corpus.py — Knowledge Corpus Ingestion & Normalization

Parses raw data sources:
1. data/source/yoga-api.json (43 poses with structured contraindications)
2. data/source/final_asan1_1.csv (226 asanas with descriptions, levels, contraindications)
3. Preserves foundational Phase 1 chunks (philosophy, pranayama, sequence guidelines)

Outputs a unified, standardized corpus to data/knowledge/corpus.json.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = REPO_ROOT / "data" / "source"
KNOWLEDGE_DIR = REPO_ROOT / "data" / "knowledge"
CORPUS_OUTPUT = KNOWLEDGE_DIR / "corpus.json"

# Common contraindication keyword extractors
KEYWORD_PATTERNS = {
    "back injury": r"back|spine|spinal|slip[ -]disc|herniat|lumbar|sacral|sciatica",
    "neck injury": r"neck|cervical",
    "knee injury": r"knee|meniscus",
    "shoulder injury": r"shoulder|rotator cuff",
    "wrist injury": r"wrist|carpal tunnel",
    "hip injury": r"hip|groin",
    "hypertension": r"high blood pressure|hypertension",
    "low blood pressure": r"low blood pressure|hypotension",
    "heart condition": r"heart|cardiac|angina",
    "pregnancy": r"pregnan|trimester",
    "menstruation": r"menstrua|period",
    "hernia": r"hernia|abdominal wall",
    "glaucoma": r"glaucoma|eye pressure",
    "vertigo": r"vertigo|dizziness|faint",
    "recent surgery": r"surgery|operation",
    "asthma": r"asthma|respiratory",
}


def _extract_keywords(text: str) -> list[str]:
    """Extract standard contraindication tags from descriptive text."""
    if not text or not isinstance(text, str):
        return []
    tags = []
    text_lower = text.lower()
    for tag, pattern in KEYWORD_PATTERNS.items():
        if re.search(pattern, text_lower):
            tags.append(tag)
    return tags


def _clean_str(val: Any) -> str:
    if val is None or (isinstance(val, float) and str(val) == "nan"):
        return ""
    return str(val).strip()


def _normalize_level(raw_level: str) -> str:
    l_lower = raw_level.lower()
    if "adv" in l_lower:
        return "Advanced"
    if "inter" in l_lower:
        return "Intermediate"
    return "Beginner"


def ingest_corpus() -> list[dict[str, Any]]:
    unified_chunks: list[dict[str, Any]] = []
    seen_sanskrit: set[str] = set()

    # 1. Retain Phase 1 foundational canonical chunks
    import subprocess
    try:
        git_res = subprocess.run(
            ["git", "show", "HEAD:data/knowledge/corpus.json"],
            capture_output=True,
            text=True,
            check=True,
        )
        p1_chunks = json.loads(git_res.stdout)
        for chunk in p1_chunks:
            chunk["contraindications"] = chunk.get("contraindications", [])
            chunk["level"] = chunk.get("level", "Beginner")
            unified_chunks.append(chunk)
            if chunk.get("title"):
                seen_sanskrit.add(chunk["title"].lower())
    except Exception as e:
        print(f"Notice: using current file fallback: {e}")

    # 2. Ingest yoga-api.json (43 poses with high-structure metadata)
    api_path = SOURCE_DIR / "yoga-api.json"
    if api_path.exists():
        with open(api_path, encoding="utf-8") as f:
            api_poses = json.load(f)
        for pose in api_poses:
            s_name = _clean_str(pose.get("sanskrit_name"))
            if not s_name:
                continue
            e_name = _clean_str(pose.get("english_name"))
            cid = f"asana_api_{len(unified_chunks) + 1:03d}"
            seen_sanskrit.add(s_name.lower())

            raw_contra = pose.get("contraindications") or []
            contra_list = []
            if isinstance(raw_contra, list):
                for c in raw_contra:
                    c_clean = str(c).strip()
                    if c_clean:
                        extracted = _extract_keywords(c_clean)
                        if extracted:
                            contra_list.extend(extracted)
                        else:
                            contra_list.append(c_clean.lower())
            contra_list = sorted(list(set(contra_list)))

            procedure_steps = pose.get("procedure") or []
            if isinstance(procedure_steps, list):
                proc_text = " ".join([str(s).strip() for s in procedure_steps])
            else:
                proc_text = str(procedure_steps)

            benefits_list = pose.get("benefits") or []
            benefits_text = " ".join([str(b).strip() for b in benefits_list]) if isinstance(benefits_list, list) else str(benefits_list)
            target_parts = pose.get("target_body_parts") or []

            title = f"{e_name} ({s_name})" if e_name else s_name
            content = (
                f"{title}. Targets: {', '.join(target_parts)}. "
                f"Instructions: {proc_text} "
                f"Benefits: {benefits_text} "
                f"Contraindications & Precautions: {', '.join(contra_list) if contra_list else 'None reported for general practice'}."
            )

            unified_chunks.append({
                "document_id": cid,
                "chunk_id": f"{cid}_001",
                "title": title,
                "sanskrit_name": s_name,
                "english_name": e_name,
                "section": "Asana Guide",
                "topic": "asana_pose",
                "level": "Beginner" if "beginner" in proc_text.lower() else "Intermediate",
                "target_body_parts": target_parts,
                "contraindications": contra_list,
                "benefits": benefits_list if isinstance(benefits_list, list) else [benefits_text],
                "content": content,
                "source_type": "curated_dataset",
                "license": "open",
            })

    # 3. Ingest final_asan1_1.csv (226 asanas)
    csv_path = SOURCE_DIR / "final_asan1_1.csv"
    if csv_path.exists():
        with open(csv_path, encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                aname = _clean_str(row.get("AName"))
                if not aname:
                    continue
                if aname.lower() in seen_sanskrit:
                    continue
                seen_sanskrit.add(aname.lower())

                cid = f"asana_csv_{len(unified_chunks) + 1:03d}"
                desc = _clean_str(row.get("Description"))
                benefits = _clean_str(row.get("Benefits"))
                raw_contra = _clean_str(row.get("Contraindications"))
                breathing = _clean_str(row.get("Breathing"))
                level = _normalize_level(_clean_str(row.get("Level")))

                contra_list = _extract_keywords(raw_contra)
                if not contra_list and raw_contra:
                    contra_list = [c.strip().lower() for c in re.split(r"[,;.]", raw_contra) if len(c.strip()) > 3][:3]

                title = aname
                content = (
                    f"{aname} ({level} level). Description: {desc} "
                    f"Benefits: {benefits} "
                    f"Breathing: {breathing} "
                    f"Contraindications: {raw_contra if raw_contra else 'None reported for gentle practice.'}"
                )

                unified_chunks.append({
                    "document_id": cid,
                    "chunk_id": f"{cid}_001",
                    "title": title,
                    "sanskrit_name": aname,
                    "english_name": aname,
                    "section": "Asana Guide",
                    "topic": "asana_pose",
                    "level": level,
                    "target_body_parts": [],
                    "contraindications": contra_list,
                    "contraindication_details": raw_contra,
                    "benefits": [b.strip() for b in benefits.split(".") if b.strip()][:4],
                    "breathing": breathing,
                    "content": content,
                    "source_type": "hatha_yoga_dataset",
                    "license": "open",
                })

    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CORPUS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(unified_chunks, f, indent=2, ensure_ascii=False)

    print(f"✅ Ingested {len(unified_chunks)} chunks into {CORPUS_OUTPUT}")
    return unified_chunks


if __name__ == "__main__":
    ingest_corpus()
