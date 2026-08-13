#!/usr/bin/env python3
"""Generate and validate NinePass daily diagnostic + English packs.

Design goals:
- No paid API or external AI dependency.
- Diagnostic questions are source-grounded composites built only from already-verified
  questions in the repository.
- English prefers never-used items from repository banks; when exhausted it rotates
  the least-recently-used verified items (spaced review) instead of inventing facts.
- Idempotent for the same target date.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import glob
import hashlib
import itertools
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SUBJECTS = ["국어", "영어", "한국사", "행정법총론", "행정학개론"]
MARKERS = ["가", "나", "다", "라", "마", "바"]
MARKER_START_RE = re.compile(r"\n\s*\(가\)\s*")
SUBQ_RE = re.compile(
    r"\((가|나|다|라|마|바)\)\s*(.*?)(?=\n\s*\((?:가|나|다|라|마|바)\)\s*|\Z)",
    re.S,
)
DATED_ENGLISH_RE = re.compile(r"daily_english_(\d{4})_(\d{2})_(\d{2})\.json$")
DATED_DIAG_RE = re.compile(r"daily_general_admin_(\d{4})_(\d{2})_(\d{2})\.json$")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def stable_hash(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def parse_date_from_filename(path: Path, regex: re.Pattern[str]) -> str | None:
    m = regex.search(path.name)
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"


def choice_tokens(text: str) -> set[str]:
    return {t.casefold() for t in re.findall(r"[가-힣A-Za-z0-9·]+", text or "") if len(t) >= 2}


def correct_choice_text(q: dict[str, Any]) -> str | None:
    """Infer the answer from explanation evidence first, encoded index second.

    Some historical packs contain an incorrect numeric answer index while the
    explanation states the correct fact. We therefore require explanation
    support for source atoms and use the uniquely best-supported choice.
    """
    choices = q.get("choices")
    answer = q.get("answer")
    explanation = str(q.get("explanation", "")).strip()
    if not isinstance(choices, list) or not choices:
        if isinstance(answer, str) and answer.strip():
            return answer.strip()
        return None

    choices_s = [str(x).strip() for x in choices]
    exp_norm = norm(explanation)
    exp_tokens = choice_tokens(explanation)
    scores: list[float] = []
    for c in choices_s:
        cn = norm(c)
        toks = choice_tokens(c)
        score = 0.0
        if cn and cn in exp_norm:
            score += 100.0
        if toks:
            overlap = toks & exp_tokens
            score += sum(max(1, len(t)) for t in overlap) / max(1, sum(max(1, len(t)) for t in toks)) * 10.0
        scores.append(score)

    best = max(scores) if scores else 0.0
    winners = [i for i, v in enumerate(scores) if v == best and v > 0]
    if len(winners) == 1:
        return choices_s[winners[0]]

    encoded_idx = None
    if isinstance(answer, int):
        if 0 <= answer < len(choices_s):
            encoded_idx = answer
        elif 1 <= answer <= len(choices_s):
            encoded_idx = answer - 1
    elif isinstance(answer, str) and answer.strip().isdigit():
        n = int(answer.strip())
        if 0 <= n < len(choices_s):
            encoded_idx = n
        elif 1 <= n <= len(choices_s):
            encoded_idx = n - 1
    if encoded_idx is not None and scores[encoded_idx] > 0:
        return choices_s[encoded_idx]

    # No explanation-supported answer => do not trust this source atom.
    return None

def split_subquestions(question: str) -> list[str]:
    m = MARKER_START_RE.search(question)
    if not m:
        return []
    tail = question[m.start() + 1 :]
    return [x.group(2).strip() for x in SUBQ_RE.finditer(tail)]


def extract_atoms_from_question(q: dict[str, Any]) -> list[dict[str, str]]:
    subject = str(q.get("subject", "")).strip()
    if subject not in SUBJECTS:
        return []
    question = str(q.get("question", "")).strip()
    if not question:
        return []
    correct = correct_choice_text(q)
    if not correct:
        return []

    subs = split_subquestions(question)
    if subs:
        answers = [x.strip() for x in re.split(r"\s+/\s+", correct)]
        if len(answers) == len(subs) and all(answers):
            src = q.get("sourceQuestionIds") or []
            atoms = []
            for i, (prompt, answer) in enumerate(zip(subs, answers)):
                if "/" in answer:
                    continue
                source_id = (
                    str(src[i])
                    if isinstance(src, list) and len(src) == len(subs)
                    else f"atom_{stable_hash(subject, prompt, answer)[:20]}"
                )
                atoms.append(
                    {
                        "subject": subject,
                        "prompt": prompt,
                        "answer": answer,
                        "source_id": source_id,
                    }
                )
            if atoms:
                return atoms

    # Single verified question fallback.
    if "/" in correct:
        return []
    source_id = str(q.get("id") or f"atom_{stable_hash(subject, question, correct)[:20]}")
    return [{"subject": subject, "prompt": question, "answer": correct, "source_id": source_id}]


def iter_question_docs(root: Path):
    for path in sorted(root.glob("*.json")):
        if path.name in {"manifest.json", "AUTO_VALIDATION_REPORT.json"}:
            continue
        if "VALIDATION_REPORT" in path.name.upper():
            continue
        try:
            data = load_json(path)
        except Exception:
            continue
        if isinstance(data, dict) and isinstance(data.get("questions"), list):
            yield path, data


def build_diagnostic_source(root: Path):
    # Use only original single questions as atoms. Composite daily questions are
    # used only to record already-used source combinations, never as truth sources.
    candidates: dict[str, dict[str, list[dict[str, str]]]] = {s: defaultdict(list) for s in SUBJECTS}
    historical_signatures: set[tuple[str, ...]] = set()
    historical_questions: set[str] = set()

    for _, data in iter_question_docs(root):
        for q in data.get("questions", []):
            if not isinstance(q, dict):
                continue
            nq = norm(q.get("question"))
            if nq:
                historical_questions.add(nq)
            src = q.get("sourceQuestionIds")
            if isinstance(src, list) and len(src) >= 2:
                historical_signatures.add(tuple(sorted(map(str, src))))

            # Never propagate a composite's possibly wrong answer into the atom bank.
            if split_subquestions(str(q.get("question", ""))):
                continue
            atoms = extract_atoms_from_question(q)
            for atom in atoms:
                candidates[atom["subject"]][norm(atom["prompt"])].append(atom)

    pools: dict[str, list[dict[str, str]]] = {}
    for subject in SUBJECTS:
        safe: list[dict[str, str]] = []
        for _, variants in candidates[subject].items():
            answers = {norm(v["answer"]) for v in variants}
            # Conflicting answers for the same prompt are quarantined.
            if len(answers) != 1:
                continue
            # Prefer the shortest stable source id deterministically.
            variants.sort(key=lambda v: (len(v["source_id"]), v["source_id"]))
            safe.append(variants[0])
        pools[subject] = safe

    return pools, historical_signatures, historical_questions

def choose_combos(
    atoms: list[dict[str, str]],
    count: int,
    n_facts: int,
    date: str,
    subject: str,
    used_signatures: set[tuple[str, ...]],
) -> list[tuple[dict[str, str], ...]]:
    # Keep atom source IDs unique and favor diverse answer strings.
    by_source: dict[str, dict[str, str]] = {}
    for a in atoms:
        by_source.setdefault(a["source_id"], a)
    atoms = list(by_source.values())

    combos = []
    # For typical source pools this is small enough and gives deterministic exhaustive safety.
    for combo in itertools.combinations(atoms, n_facts):
        sig = tuple(sorted(a["source_id"] for a in combo))
        if sig in used_signatures:
            continue
        if len({norm(a["prompt"]) for a in combo}) != n_facts:
            continue
        if len({norm(a["answer"]) for a in combo}) < max(2, n_facts - 1):
            continue
        combos.append(combo)

    combos.sort(key=lambda c: stable_hash(date, subject, *(a["source_id"] for a in c)))
    return combos[:count]


def make_distractors(
    correct_parts: list[str],
    answer_pool: list[str],
    rng: random.Random,
) -> list[str]:
    correct = " / ".join(correct_parts)
    unique_pool = [x for x in dict.fromkeys(answer_pool) if norm(x) not in {norm(y) for y in correct_parts} and "/" not in x]
    if not unique_pool:
        raise RuntimeError("Not enough alternative answers for distractor generation")
    result: list[str] = []
    attempts = 0
    while len(result) < 3 and attempts < 500:
        attempts += 1
        parts = list(correct_parts)
        change_count = 1 if len(parts) == 2 else rng.choice([1, 1, 2])
        for pos in rng.sample(range(len(parts)), k=min(change_count, len(parts))):
            parts[pos] = rng.choice(unique_pool)
        candidate = " / ".join(parts)
        if norm(candidate) == norm(correct):
            continue
        if candidate not in result:
            result.append(candidate)
    if len(result) != 3:
        raise RuntimeError("Could not build three unique distractors")
    return result


def generate_diagnostic(root: Path, target_date: str) -> dict[str, Any]:
    compact = target_date.replace("-", "")
    pack_id = f"daily_general_admin_{target_date.replace('-', '_')}"
    pools, historical_signatures, historical_questions = build_diagnostic_source(root)
    questions: list[dict[str, Any]] = []
    used_signatures = set(historical_signatures)
    serial = 1

    for subject in SUBJECTS:
        atoms = pools.get(subject, [])
        if len(atoms) < 8:
            raise RuntimeError(f"{subject}: verified atomic source pool too small ({len(atoms)})")
        answer_pool = [a["answer"] for a in atoms]

        # 4 medium 2-fact composites + 4 hard 3-fact composites.
        plan = [("중", 2, 4), ("상", 3, 4)]
        for difficulty, n_facts, needed in plan:
            selected = choose_combos(atoms, needed, n_facts, target_date, subject + difficulty, used_signatures)
            if len(selected) < needed:
                # Fallback to a larger fact count rather than reusing an old source combination.
                alt_n = min(n_facts + 1, 4)
                selected = choose_combos(atoms, needed, alt_n, target_date, subject + difficulty + "fallback", used_signatures)
            if len(selected) < needed:
                raise RuntimeError(f"{subject}/{difficulty}: not enough unused verified source combinations")

            for combo in selected:
                sig = tuple(sorted(a["source_id"] for a in combo))
                used_signatures.add(sig)
                rng = random.Random(int(stable_hash(target_date, subject, str(serial))[:16], 16))
                parts = [a["answer"] for a in combo]
                correct = " / ".join(parts)
                distractors = make_distractors(parts, answer_pool, rng)
                choices = distractors + [correct]
                rng.shuffle(choices)
                answer_idx = choices.index(correct)

                subblocks = []
                for i, atom in enumerate(combo):
                    subblocks.append(f"({MARKERS[i]}) {atom['prompt']}")
                question_text = (
                    f"[{subject} 복합 진단] 다음 {len(combo)}개 물음의 답을 "
                    f"(가)부터 순서대로 바르게 짝지은 것은?\n\n" + "\n".join(subblocks)
                )
                if norm(question_text) in historical_questions:
                    raise RuntimeError("Unexpected exact historical duplicate during diagnostic generation")
                historical_questions.add(norm(question_text))

                explanation = "정답 대응은 " + ", ".join(
                    f"({MARKERS[i]}) {a['answer']}" for i, a in enumerate(combo)
                ) + "이다."

                questions.append(
                    {
                        "id": f"nga_{compact}_{serial:03d}",
                        "examType": "9급 일반행정",
                        "stage": "balanced_daily",
                        "subject": subject,
                        "domain": "복합 진단",
                        "subdomain": "원천 검증형",
                        "difficulty": difficulty,
                        "question": question_text,
                        "choices": choices,
                        "answer": answer_idx,
                        "explanation": explanation,
                        "sourceBasis": "설명-정답 교차검증을 통과한 원본 단일 문항만 재조합한 source-grounded 복합 진단 문항",
                        "estimatedTimeSec": 100 if len(combo) == 2 else 130,
                        "date": target_date,
                        "packId": pack_id,
                        "sourceQuestionIds": [a["source_id"] for a in combo],
                    }
                )
                serial += 1

    return {
        "schemaVersion": 1,
        "packId": pack_id,
        "date": target_date,
        "title": f"{target_date} 9급 일반행정 균형 진단 40문제",
        "description": "국어·영어·한국사·행정법총론·행정학개론 각 8문항",
        "track": "general_admin_9grade",
        "personalizationStatus": "daily_balanced_source_grounded_auto",
        "personalization": {"subjects": SUBJECTS, "questionsPerSubject": 8},
        "questions": questions,
    }


def scan_english_candidates(root: Path):
    vocab_candidates: dict[str, dict[str, Any]] = {}
    sent_candidates: dict[str, dict[str, Any]] = {}
    vocab_last_seen: dict[str, str] = {}
    sent_last_seen: dict[str, str] = {}

    for path in sorted(root.glob("*.json")):
        if path.name == "daily_english_latest.json":
            continue
        try:
            data = load_json(path)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue

        dated = parse_date_from_filename(path, DATED_ENGLISH_RE)
        vocabulary = data.get("vocabulary")
        if isinstance(vocabulary, list):
            for item in vocabulary:
                if not isinstance(item, dict):
                    continue
                word = str(item.get("word", "")).strip()
                meaning = str(item.get("meaning", "")).strip()
                if not word or not meaning:
                    continue
                key = norm(word)
                vocab_candidates.setdefault(key, copy.deepcopy(item))
                if dated:
                    vocab_last_seen[key] = max(vocab_last_seen.get(key, ""), dated)

        sentences = data.get("sentences")
        if isinstance(sentences, list):
            for item in sentences:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text", "")).strip()
                translation = str(item.get("translation", "")).strip()
                if not text or not translation:
                    continue
                key = norm(text)
                sent_candidates.setdefault(key, copy.deepcopy(item))
                if dated:
                    sent_last_seen[key] = max(sent_last_seen.get(key, ""), dated)

    return vocab_candidates, sent_candidates, vocab_last_seen, sent_last_seen


def pick_rotation(keys: list[str], last_seen: dict[str, str], n: int, target_date: str, salt: str) -> list[str]:
    # Never-used bank items first, then oldest last-seen. Stable hash breaks ties without randomness drift.
    ranked = sorted(
        keys,
        key=lambda k: (
            1 if k in last_seen else 0,
            last_seen.get(k, "0000-00-00"),
            stable_hash(target_date, salt, k),
        ),
    )
    if len(ranked) < n:
        raise RuntimeError(f"English verified source pool too small: need {n}, found {len(ranked)}")
    return ranked[:n]


def generate_english(root: Path, target_date: str) -> dict[str, Any]:
    compact = target_date.replace("-", "")
    pack_id = f"daily_english_{target_date.replace('-', '_')}"
    vocab_map, sent_map, vocab_seen, sent_seen = scan_english_candidates(root)
    vocab_keys = pick_rotation(list(vocab_map), vocab_seen, 20, target_date, "vocab")
    sent_keys = pick_rotation(list(sent_map), sent_seen, 10, target_date, "sentence")

    vocabulary = []
    for i, key in enumerate(vocab_keys, 1):
        src = copy.deepcopy(vocab_map[key])
        src["id"] = f"eng_vocab_{compact}_{i:03d}"
        src["priority"] = src.get("priority") or "A"
        src["sourceBasis"] = "저장소 누적 검증 어휘 기반 자동 선택(미사용 우선·이후 간격 복습)"
        vocabulary.append(src)

    sentences = []
    for i, key in enumerate(sent_keys, 1):
        src = copy.deepcopy(sent_map[key])
        src["id"] = f"eng_sentence_{compact}_{i:03d}"
        src["speechText"] = src.get("speechText") or src.get("text")
        src["sourceBasis"] = "저장소 누적 검증 문장 기반 자동 선택(미사용 우선·이후 간격 복습)"
        sentences.append(src)

    rng = random.Random(int(stable_hash(target_date, "english_quiz")[:16], 16))
    quiz = []
    meanings = [v["meaning"] for v in vocabulary]
    for i, v in enumerate(vocabulary, 1):
        correct = v["meaning"]
        distractor_pool = list(dict.fromkeys(m for m in meanings if norm(m) != norm(correct)))
        if len(distractor_pool) < 3:
            raise RuntimeError("Not enough distinct English meanings for quiz distractors")
        choices = rng.sample(distractor_pool, 3) + [correct]
        rng.shuffle(choices)
        quiz.append(
            {
                "id": f"eng_quiz_{compact}_{i:03d}",
                "type": "vocabulary",
                "prompt": f"다음 문장에서 '{v['word']}'의 뜻으로 가장 적절한 것은?\n{v.get('example', '')}",
                "choices": choices,
                "answer": choices.index(correct),
                "explanation": f"'{v['word']}'는 '{correct}'라는 뜻이다.",
                "relatedItemId": v["id"],
            }
        )

    translations = [s["translation"] for s in sentences]
    for j, s in enumerate(sentences, 1):
        correct = s["translation"]
        distractor_pool = list(dict.fromkeys(t for t in translations if norm(t) != norm(correct)))
        if len(distractor_pool) < 3:
            raise RuntimeError("Not enough distinct sentence translations for quiz distractors")
        choices = rng.sample(distractor_pool, 3) + [correct]
        rng.shuffle(choices)
        quiz.append(
            {
                "id": f"eng_quiz_{compact}_{20 + j:03d}",
                "type": "sentence",
                "prompt": f"다음 문장의 우리말 뜻으로 가장 적절한 것은?\n{s['text']}",
                "choices": choices,
                "answer": choices.index(correct),
                "explanation": correct,
                "relatedItemId": s["id"],
            }
        )

    unseen_vocab = sum(1 for k in vocab_keys if k not in vocab_seen)
    unseen_sent = sum(1 for k in sent_keys if k not in sent_seen)
    mode = "new_from_verified_bank" if unseen_vocab == 20 and unseen_sent == 10 else "verified_bank_spaced_review"

    return {
        "schemaVersion": 1,
        "packId": pack_id,
        "date": target_date,
        "title": f"{target_date} 취약 영어 자동 학습·퀴즈팩",
        "description": f"검증 어휘 20개, 문장 10개, 연계 퀴즈 30개 · mode={mode}",
        "generationMode": mode,
        "newVocabularyCount": unseen_vocab,
        "newSentenceCount": unseen_sent,
        "vocabulary": vocabulary,
        "sentences": sentences,
        "quizQuestions": quiz,
    }


def update_manifest(root: Path, target_date: str, diag: dict[str, Any], eng: dict[str, Any]) -> dict[str, Any]:
    path = root / "manifest.json"
    manifest = load_json(path)
    manifest["updatedAt"] = f"{target_date}T06:00:00+09:00"
    manifest.setdefault("schemaVersion", 1)
    manifest.setdefault("app", "NinePass")
    manifest.setdefault("track", "general_admin_9grade")
    manifest.setdefault("replacePacks", False)
    packs = manifest.setdefault("packs", [])

    diag_file = f"daily_general_admin_{target_date.replace('-', '_')}.json"
    eng_file = f"daily_english_{target_date.replace('-', '_')}.json"
    diag_entry = {
        "id": diag["packId"],
        "file": diag_file,
        "packType": "diagnostic",
        "enabled": True,
        "questionCount": 40,
        "date": target_date,
    }
    eng_entry = {
        "id": f"daily-english-{target_date}",
        "file": eng_file,
        "packType": "english",
        "enabled": True,
        "questionCount": 30,
        "vocabularyCount": 20,
        "sentenceCount": 10,
        "quizQuestionCount": 30,
        "date": target_date,
    }

    def upsert(entry: dict[str, Any], matcher):
        for i, old in enumerate(packs):
            if matcher(old):
                packs[i] = entry
                return
        packs.append(entry)

    upsert(diag_entry, lambda x: isinstance(x, dict) and (x.get("date") == target_date and x.get("packType") == "diagnostic"))
    upsert(eng_entry, lambda x: isinstance(x, dict) and (x.get("date") == target_date and x.get("packType") == "english"))
    return manifest


def validate(diag: dict[str, Any], eng: dict[str, Any], manifest: dict[str, Any], target_date: str) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def ck(name: str, ok: bool, detail: Any = None):
        checks.append({"name": name, "pass": bool(ok), "detail": detail})

    q = diag.get("questions", [])
    ck("diagnostic_total_40", len(q) == 40, len(q))
    counts = Counter(x.get("subject") for x in q)
    ck("diagnostic_5_subjects_x8", all(counts.get(s) == 8 for s in SUBJECTS), dict(counts))
    ck("diagnostic_unique_ids", len({x.get("id") for x in q}) == 40)
    ck("diagnostic_unique_questions", len({norm(x.get("question")) for x in q}) == 40)
    ck("diagnostic_choices_answer_valid", all(isinstance(x.get("choices"), list) and len(x["choices"]) == 4 and isinstance(x.get("answer"), int) and 0 <= x["answer"] < 4 for x in q))
    ck("diagnostic_source_grounded", all(isinstance(x.get("sourceQuestionIds"), list) and len(x["sourceQuestionIds"]) >= 2 for x in q))

    vocab = eng.get("vocabulary", [])
    sent = eng.get("sentences", [])
    quiz = eng.get("quizQuestions", [])
    ck("english_vocab_20", len(vocab) == 20, len(vocab))
    ck("english_sentences_10", len(sent) == 10, len(sent))
    ck("english_quiz_30", len(quiz) == 30, len(quiz))
    ck("english_vocab_unique", len({norm(x.get("word")) for x in vocab}) == 20)
    ck("english_sentence_unique", len({norm(x.get("text")) for x in sent}) == 10)
    ck("english_quiz_answers_valid", all(isinstance(x.get("choices"), list) and len(x["choices"]) == 4 and isinstance(x.get("answer"), int) and 0 <= x["answer"] < 4 for x in quiz))

    packs = manifest.get("packs", [])
    ck("manifest_updated_date", manifest.get("updatedAt") == f"{target_date}T06:00:00+09:00", manifest.get("updatedAt"))
    ck("manifest_has_diagnostic", any(isinstance(x, dict) and x.get("date") == target_date and x.get("packType") == "diagnostic" for x in packs))
    ck("manifest_has_english", any(isinstance(x, dict) and x.get("date") == target_date and x.get("packType") == "english" for x in packs))

    passed = sum(1 for x in checks if x["pass"])
    return {
        "schemaVersion": 1,
        "date": target_date,
        "generator": "nine-grade-source-grounded-auto-v1",
        "status": "PASS" if passed == len(checks) else "FAIL",
        "passed": passed,
        "total": len(checks),
        "checks": checks,
        "englishGenerationMode": eng.get("generationMode"),
        "newVocabularyCount": eng.get("newVocabularyCount"),
        "newSentenceCount": eng.get("newSentenceCount"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="Target date YYYY-MM-DD in Asia/Seoul")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    try:
        dt.date.fromisoformat(args.date)
    except ValueError:
        print("Invalid --date; expected YYYY-MM-DD", file=sys.stderr)
        return 2

    root = Path(args.root).resolve()
    target_date = args.date
    diag_path = root / f"daily_general_admin_{target_date.replace('-', '_')}.json"
    eng_path = root / f"daily_english_{target_date.replace('-', '_')}.json"
    latest_path = root / "daily_english_latest.json"

    # If same-day dated packs already exist, preserve them exactly and only repair manifest/latest/report.
    if diag_path.exists() and eng_path.exists():
        diag = load_json(diag_path)
        eng = load_json(eng_path)
        print(f"{target_date}: dated packs already exist; preserving existing content")
    else:
        diag = generate_diagnostic(root, target_date)
        eng = generate_english(root, target_date)
        dump_json(diag_path, diag)
        dump_json(eng_path, eng)

    dump_json(latest_path, eng)
    manifest = update_manifest(root, target_date, diag, eng)
    dump_json(root / "manifest.json", manifest)
    report = validate(diag, eng, manifest, target_date)
    dump_json(root / "AUTO_VALIDATION_REPORT.json", report)

    if report["status"] != "PASS":
        print(json.dumps(report, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    print(f"PASS {report['passed']}/{report['total']} | diagnostic=40 | English=20/10/30 | mode={eng.get('generationMode')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
