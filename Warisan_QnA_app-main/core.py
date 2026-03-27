# Core QnA processing functions - shared by CLI and GUI

from __future__ import annotations
import os
import json
import difflib
import re
import requests
import dotenv
from typing import Any, List, Dict, Tuple, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from dotenv import load_dotenv

load_dotenv(override=True)

# --- Configuration ---
MAX_PAIRS = int(os.getenv("QNA_MAX_PAIRS", "100"))
CHUNK_WORDS = int(os.getenv("QNA_CHUNK_WORDS", "800"))
CHUNK_OVERLAP = int(os.getenv("QNA_CHUNK_OVERLAP", "100"))
SIM_THRESH = float(os.getenv("QNA_DUP_QUESTION_SIM", "0.88"))

# --- Ollama remote server configuration ---
# Native Ollama /api/generate endpoint on the LAN server
OLLAMA_URL   = os.getenv("OLLAMA_URL",   "https://arrow-test-harley-chorus.trycloudflare.com/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

# Keep these for compatibility with health-check route in web.py
API_KEY  = os.getenv("OPENAI_API_KEY", "ollama")
BASE_URL = os.getenv("OPENAI_BASE_URL", OLLAMA_URL)
MODEL_GEN    = OLLAMA_MODEL
MODEL_REVIEW = OLLAMA_MODEL

# --- Load System Prompts from files ---
def load_prompt_candidates(*relative_paths: str) -> str:
    """Load prompt from first existing path in the repository root."""
    base_dir = os.path.dirname(__file__)
    for rel_path in relative_paths:
        if not rel_path:
            continue
        prompt_path = os.path.join(base_dir, rel_path)
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            continue
    return ""

PREFILTER_SYSTEM = load_prompt_candidates(
    "prompts/prefilter_system_v2.txt",
    "prompts/prefilter_system.txt",
    "new prompts/prefilter_system_v2.txt",
)
FACTEXTRACT_SYSTEM = load_prompt_candidates(
    "prompts/factextract_system.txt",
    "new prompts/factextract_system.txt",
)
GENERATOR_SYSTEM = load_prompt_candidates(
    "prompts/generator_system_v2.txt",
    "prompts/generator_system.txt",
    "new prompts/generator_system_v2.txt",
)
VARIATION_SYSTEM = load_prompt_candidates(
    "prompts/variation_system.txt",
    "new prompts/variation_system.txt",
)
REVIEWER_SYSTEM = load_prompt_candidates(
    "prompts/reviewer_system_v2.txt",
    "prompts/reviewer_system.txt",
    "new prompts/reviewer_system_v2.txt",
)

# Fallback prompts if files not found
if not PREFILTER_SYSTEM:
    PREFILTER_SYSTEM = (
        "Extract TITLE, ABSTRACT_BLOCK, SOURCE, BODY_BLOCK from FULL TEXT and "
        "return CLEAN_TEXT blocks only."
    )

if not FACTEXTRACT_SYSTEM:
    FACTEXTRACT_SYSTEM = (
        "Extract atomic facts from CLEAN_TEXT as JSONL: "
        "{\"fakta\":\"...\",\"jenis\":\"identiti|kausal|metodologi|perbandingan|warisan\",\"source\":\"...\"}."
    )

if not GENERATOR_SYSTEM:
    GENERATOR_SYSTEM = (
        "Generate Malay Q&A JSONL from FACT_LIST + CLEAN_TEXT only: "
        "{\"question\":\"...\",\"answer\":\"...\",\"curriculum_phase\":0,\"source\":\"...\"}."
    )

if not VARIATION_SYSTEM:
    VARIATION_SYSTEM = (
        "Generate exactly 2 controlled question variations per base Q&A as JSONL with "
        "jenis_variasi and curriculum_phase."
    )

if not REVIEWER_SYSTEM:
    REVIEWER_SYSTEM = (
        "Review one Q&A and return JSON: "
        "{\"status\":\"accept|edit|reject\",\"question\":\"...\",\"answer\":\"...\","
        "\"petikan_sumber\":\"\",\"curriculum_phase\":0,\"reason\":\"...\"}."
    )

# --- Chat Helper (Ollama native /api/generate) ---
def chat(model: str, system: str, user: str, temperature: float = 0.2) -> str:
    """
    Send a prompt to the remote Ollama server via its native /api/generate API.
    The system prompt and user message are combined into a single prompt string
    because Ollama's native API does not use a messages array.
    """
    # Combine system + user into one prompt (Ollama native format)
    combined_prompt = f"{system}\n\n{user}".strip() if system else user

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": combined_prompt,
        "stream": False,
    }

    print(f"[Ollama] Sending request to {OLLAMA_URL} | model={OLLAMA_MODEL} | prompt_len={len(combined_prompt)}")

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    except requests.exceptions.ConnectionError as e:
        print(f"[Ollama] Connection error: {e}")
        raise ValueError(f"Cannot reach Ollama server at {OLLAMA_URL}. Check that the server is running and reachable.")
    except requests.exceptions.Timeout:
        print(f"[Ollama] Request timed out after 120s")
        raise ValueError("Ollama request timed out (120 s). The model may be loading or overloaded.")
    except requests.exceptions.RequestException as e:
        print(f"[Ollama] Request error: {e}")
        raise ValueError(f"Ollama request failed: {e}")

    print(f"[Ollama] Response status: {resp.status_code}")

    if resp.status_code != 200:
        print(f"[Ollama] Non-200 response body: {resp.text[:500]}")
        raise ValueError(f"Ollama returned HTTP {resp.status_code}: {resp.text[:300]}")

    try:
        data = resp.json()
    except ValueError:
        print(f"[Ollama] Could not parse JSON response: {resp.text[:500]}")
        raise ValueError("Ollama returned invalid JSON.")

    content = data.get("response", "")
    if not content:
        print(f"[Ollama] Warning: 'response' field missing or empty. Full payload: {data}")
        raise ValueError("Ollama response JSON did not contain a 'response' field.")

    # Strip <think>...</think> tags emitted by reasoning models (e.g. DeepSeek-R1, Qwen3)
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    return content

# --- Text Chunking ---
def chunk_words(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> List[Tuple[str, int, int]]:
    """Chunk text by words with overlap (no limit on words or chunks)"""
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(1, size - overlap)
    # Process all words without stopping early
    i = 0
    while i < len(words):
        chunk_words_list = words[i:i + size]
        if chunk_words_list:
            chunk_text = " ".join(chunk_words_list)
            chunks.append((chunk_text, i, min(i + size, len(words))))
        i += step
        # Only stop if we've processed everything
        if i >= len(words):
            break
    return chunks

# --- Parsing helpers ---
def _extract_json_object(raw: str) -> Optional[Dict[str, Any]]:
    """Extract first JSON object from a model response."""
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    if "{" not in raw or "}" not in raw:
        return None
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start >= end:
        return None
    try:
        obj = json.loads(raw[start:end])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _parse_jsonl_objects(raw: str) -> List[Dict[str, Any]]:
    """Parse loose JSONL output; skips malformed lines."""
    results: List[Dict[str, Any]] = []
    for line in (raw or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("```"):
            continue
        obj = _extract_json_object(stripped)
        if obj:
            results.append(obj)
    return results


def _parse_clean_text_blocks(raw: str) -> Dict[str, str]:
    """Parse CLEAN_TEXT block response into title/abstract/source/body."""
    keys = {"TITLE", "ABSTRACT_BLOCK", "SOURCE", "BODY_BLOCK"}
    buckets: Dict[str, List[str]] = {k: [] for k in keys}
    current: Optional[str] = None
    for line in (raw or "").splitlines():
        stripped = line.strip()
        if stripped.upper() == "CLEAN_TEXT:":
            continue
        if stripped.endswith(":"):
            label = stripped[:-1].strip().upper()
            if label in keys:
                current = label
                continue
        if current:
            buckets[current].append(line.rstrip())
    return {
        "title": "\n".join(buckets["TITLE"]).strip(),
        "abstract": "\n".join(buckets["ABSTRACT_BLOCK"]).strip(),
        "source": "\n".join(buckets["SOURCE"]).strip(),
        "body": "\n".join(buckets["BODY_BLOCK"]).strip(),
    }


def _format_clean_text_block(
    *,
    title: str,
    abstract: str,
    body: str,
    source: str = "",
) -> str:
    """Format CLEAN_TEXT payload consistently for all stages."""
    lines = [
        "CLEAN_TEXT:",
        "TITLE:",
        (title or "").strip(),
        "",
        "ABSTRACT_BLOCK:",
        (abstract or "").strip(),
        "",
        "SOURCE:",
        (source or "").strip(),
        "",
        "BODY_BLOCK:",
        (body or "").strip(),
    ]
    return "\n".join(lines).strip()


# --- Stage 1: prefilter_v2 ---
def prefilter_document(full_text: str, source_name: str, *, title_hint: Optional[str] = None) -> Dict[str, str]:
    """Run prefilter_v2 over full text and return CLEAN_TEXT fields."""
    clean_fallback = {
        "title": (title_hint or source_name or "").strip(),
        "abstract": "",
        "source": "",
        "body": (full_text or "").strip(),
    }
    if not (full_text or "").strip():
        return clean_fallback

    prompt = f"FULL TEXT:\n{full_text.strip()}\n\nReturn CLEAN_TEXT blocks as specified."
    try:
        raw = chat(MODEL_GEN, PREFILTER_SYSTEM, prompt, temperature=0.0).strip()
    except Exception:
        return clean_fallback

    parsed = _parse_clean_text_blocks(raw)
    if not parsed["title"]:
        parsed["title"] = clean_fallback["title"]
    if not parsed["body"]:
        parsed["body"] = clean_fallback["body"]

    raw_words = len(clean_fallback["body"].split())
    body_words = len(parsed["body"].split())
    if raw_words > 0 and body_words / raw_words < 0.5:
        parsed["body"] = clean_fallback["body"]

    return parsed


# --- Stage 2: factextract ---
def extract_atomic_facts_for_chunk(
    *,
    title: str,
    abstract: str,
    body: str,
    source_label: str,
    max_facts: int = 80,
) -> List[Dict[str, str]]:
    """Extract atomic facts from CLEAN_TEXT chunk."""
    user_prompt = "\n".join([
        _format_clean_text_block(title=title, abstract=abstract, body=body),
        "",
        f"SOURCE_LABEL: {source_label}",
    ])
    try:
        raw = chat(MODEL_GEN, FACTEXTRACT_SYSTEM, user_prompt, temperature=0.0)
    except Exception as e:
        print(f"Error extracting facts: {e}")
        return []

    facts: List[Dict[str, str]] = []
    seen: set[str] = set()
    for obj in _parse_jsonl_objects(raw):
        fakta = str(obj.get("fakta") or "").strip()
        if not fakta:
            continue
        key = fakta.lower()
        if key in seen:
            continue
        seen.add(key)
        jenis = str(obj.get("jenis") or "identiti").strip().lower() or "identiti"
        src = str(obj.get("source") or source_label).strip() or source_label
        facts.append({"fakta": fakta, "jenis": jenis, "source": src})
        if len(facts) >= max_facts:
            break
    return facts


# --- Stage 3: generator_v2 ---
def generate_pairs_for_chunk(
    *,
    fact_list: List[Dict[str, str]],
    title: str,
    abstract: str,
    body: str,
    source_label: str,
    curriculum_phase: int,
    cap_this_chunk: Optional[int] = None,
    total_target: Optional[int] = None,
    produced_so_far: Optional[int] = None,
    remaining_chunks: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Generate base Q&A pairs from atomic facts + CLEAN_TEXT."""
    if not fact_list:
        return []
    user_lines: List[str] = [
        "FACT_LIST:",
        json.dumps(fact_list, ensure_ascii=False),
        "",
        _format_clean_text_block(title=title, abstract=abstract, body=body),
        "",
        f"SOURCE_LABEL: {source_label}",
        f"CURRICULUM_PHASE: {int(max(0, min(5, curriculum_phase)))}",
    ]
    if total_target is not None:
        user_lines.append(f"MIN_TARGET = 80, TOTAL_TARGET = {int(total_target)}")
    else:
        user_lines.append("MIN_TARGET = 80, TOTAL_TARGET = 100")
    if produced_so_far is not None:
        user_lines.append(f"PRODUCED_SO_FAR = {int(max(0, produced_so_far))}")
    if remaining_chunks is not None:
        user_lines.append(f"REMAINING_CHUNKS = {int(max(0, remaining_chunks))}")
    if cap_this_chunk is not None:
        user_lines.append(f"CAP_THIS_CHUNK = {int(max(0, cap_this_chunk))}")
    user_prompt = "\n".join(user_lines)

    try:
        raw = chat(MODEL_GEN, GENERATOR_SYSTEM, user_prompt, temperature=0.2)
    except Exception as e:
        print(f"Error generating pairs: {e}")
        return []

    pairs: List[Dict[str, Any]] = []
    for obj in _parse_jsonl_objects(raw):
        q = str(obj.get("question") or "").strip()
        a = str(obj.get("answer") or "").strip()
        if not q or not a:
            continue
        phase_raw = obj.get("curriculum_phase", curriculum_phase)
        try:
            phase = int(phase_raw)
        except (ValueError, TypeError):
            phase = int(curriculum_phase)
        phase = max(0, min(5, phase))
        pair = {
            "question": q,
            "answer": a,
            "curriculum_phase": phase,
            "source": str(obj.get("source") or source_label).strip() or source_label,
        }
        petikan = str(obj.get("petikan_sumber") or "").strip()
        if petikan:
            pair["petikan_sumber"] = petikan
        pairs.append(pair)
    if cap_this_chunk is not None and cap_this_chunk >= 0:
        return pairs[:cap_this_chunk]
    return pairs


# --- Stage 4: variation ---
def generate_variations_for_pairs(
    *,
    base_pairs: List[Dict[str, Any]],
    title: str,
    abstract: str,
    body: str,
) -> List[Dict[str, Any]]:
    """Generate controlled question variations for base pairs."""
    if not base_pairs:
        return []
    payload = [
        {
            "question": p.get("question", ""),
            "answer": p.get("answer", ""),
            "curriculum_phase": p.get("curriculum_phase", 1),
            "source": p.get("source", ""),
        }
        for p in base_pairs
    ]
    user_prompt = "\n".join([
        "BASE_QA:",
        json.dumps(payload, ensure_ascii=False),
        "",
        _format_clean_text_block(title=title, abstract=abstract, body=body),
    ])
    try:
        raw = chat(MODEL_GEN, VARIATION_SYSTEM, user_prompt, temperature=0.2)
    except Exception as e:
        print(f"Error generating variations: {e}")
        return []

    variations: List[Dict[str, Any]] = []
    for obj in _parse_jsonl_objects(raw):
        q = str(obj.get("question") or "").strip()
        a = str(obj.get("answer") or "").strip()
        if not q or not a:
            continue
        try:
            phase = int(obj.get("curriculum_phase", 1))
        except (ValueError, TypeError):
            phase = 1
        phase = max(0, min(5, phase))
        entry = {
            "question": q,
            "answer": a,
            "jenis_variasi": str(obj.get("jenis_variasi") or "").strip(),
            "curriculum_phase": phase,
            "source": str(obj.get("source") or "").strip(),
        }
        if not entry["source"]:
            continue
        variations.append(entry)
    return variations


# --- Stage 5: reviewer_v2 ---
def review_pair(
    pair: Dict[str, Any],
    supporting_text: str,
    *,
    title: Optional[str] = None,
    abstract: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Review one Q&A pair against its supporting chunk."""
    phase = pair.get("curriculum_phase", 1)
    try:
        phase = int(phase)
    except (ValueError, TypeError):
        phase = 1
    phase = max(0, min(5, phase))

    pair_payload: Dict[str, Any] = {
        "question": pair.get("question", ""),
        "answer": pair.get("answer", ""),
        "curriculum_phase": phase,
        "source": pair.get("source", ""),
    }
    if pair.get("petikan_sumber"):
        pair_payload["petikan_sumber"] = pair.get("petikan_sumber")

    user_lines: List[str] = [
        _format_clean_text_block(
            title=(title or "").strip(),
            abstract=(abstract or "").strip(),
            body=supporting_text.strip(),
        ),
        "",
        f"SOURCE_LABEL: {pair.get('source', '')}",
        f"CURRICULUM_PHASE: {phase}",
        "",
        "PAIR:",
        json.dumps(pair_payload, ensure_ascii=False),
    ]
    review_prompt = "\n".join(user_lines)

    raw = chat(MODEL_REVIEW, REVIEWER_SYSTEM, review_prompt, temperature=0.0).strip()
    obj = _extract_json_object(raw)
    if not obj:
        return None, "cannot_parse_reviewer"

    status = str(obj.get("status") or "").lower().strip()
    if status == "reject":
        return None, str(obj.get("reason") or "rejected")

    if status in {"accept", "edit"}:
        q = str(obj.get("question") or pair.get("question") or "").strip()
        a = str(obj.get("answer") or pair.get("answer") or "").strip()
        if q and a:
            reviewed: Dict[str, Any] = {
                "question": q,
                "answer": a,
                "source": pair.get("source"),
                "curriculum_phase": phase,
            }
            petikan = str(obj.get("petikan_sumber") or pair.get("petikan_sumber") or "").strip()
            if petikan:
                reviewed["petikan_sumber"] = petikan
            return reviewed, None
    return None, f"invalid_status: {status}"

# --- Deduplication by fuzzy similarity ---
def is_dup_question(question: str, existing_questions: List[str], threshold: float = SIM_THRESH) -> bool:
    """Check if question is a near-duplicate using fuzzy matching"""
    q_lower = question.lower().strip()
    for existing_q in existing_questions:
        existing_lower = existing_q.lower().strip()
        similarity = difflib.SequenceMatcher(None, q_lower, existing_lower).ratio()
        if similarity >= threshold:
            return True
    return False

# --- Process single text file with async/parallel processing ---
def process_text_file(text_content: str, source_name: str, max_pairs: Optional[int] = None, 
                     progress_callback: Optional[Callable[[str], None]] = None,
                     max_workers: int = 5,
                     skip_review: bool = False,
                     doc_title: Optional[str] = None) -> List[Dict]:
    """Process a single text file and return Q&A pairs using parallel processing"""
    accepted_pairs = []
    existing_questions = []
    lock = Lock()  # For thread-safe access to shared data
    
    if progress_callback:
        progress_callback(f"Processing: {source_name}")

    # Stage 1: Raw Text -> prefilter_v2
    if progress_callback:
        progress_callback("Stage prefilter_v2: extracting and canonicalizing CLEAN_TEXT")
    clean_doc = prefilter_document(
        text_content,
        source_name,
        title_hint=doc_title or source_name,
    )
    resolved_title = (clean_doc.get("title") or doc_title or source_name).strip()
    resolved_abstract = (clean_doc.get("abstract") or "").strip()
    resolved_source = (clean_doc.get("source") or "").strip()
    clean_body = (clean_doc.get("body") or text_content or "").strip()
    if progress_callback:
        body_wc = len(clean_body.split())
        progress_callback(
            f"Prefilter complete | title='{resolved_title[:80]}' | "
            f"abstract_words={len(resolved_abstract.split())} | body_words={body_wc}"
        )

    chunks = chunk_words(clean_body, CHUNK_WORDS, CHUNK_OVERLAP)
    total_chunks = len(chunks)
    
    if total_chunks == 0:
        if progress_callback:
            progress_callback("No chunks generated from text")
        return []
    
    # Adaptive max_pairs based on document size (always calculated)
    word_count = len(clean_body.split())
    # Estimate: ~15-20 pairs per 800-word chunk, but cap at reasonable limits
    estimated_pairs = min(word_count // 40, total_chunks * 20)  # 1 pair per ~40 words or 20 per chunk
    # Ensure minimum of 50 and maximum of 200
    adaptive_max = max(50, min(200, estimated_pairs))
    # Round to nearest 10
    adaptive_max = round(adaptive_max / 10) * 10
    
    # Apply user cap if provided (max_pairs is used as a cap, not absolute value)
    # If max_pairs is None/0/negative, use adaptive only
    if max_pairs is not None and max_pairs > 0:
        if max_pairs < adaptive_max:
            final_max = max_pairs
            if progress_callback:
                progress_callback(f"Adaptive max_pairs: {adaptive_max}, capped at {final_max} by user (based on {word_count} words, {total_chunks} chunks)")
        else:
            final_max = adaptive_max
            if progress_callback:
                progress_callback(f"Adaptive max_pairs set to {final_max} (user cap {max_pairs} not limiting, based on {word_count} words, {total_chunks} chunks)")
    else:
        final_max = adaptive_max
        if progress_callback:
            progress_callback(f"Adaptive max_pairs set to {final_max} based on {word_count} words and {total_chunks} chunks")
    
    max_pairs = final_max  # Update for use in rest of function
    
    if progress_callback:
        progress_callback(f"Found {total_chunks} chunks. Target: {max_pairs} pairs. Processing in parallel...")
    
    def process_chunk(chunk_data: Tuple[str, str, int, int]) -> List[Dict]:
        """Process a single chunk and return reviewed pairs"""
        chunk_text, src_name, idx, total = chunk_data
        chunk_results = []
        
        try:
            source_label = f"{src_name} Chunk {idx}"
            # Allocate chunk budget from remaining global target.
            with lock:
                current_produced = len(accepted_pairs)
            remaining_budget = max(0, max_pairs - current_produced)
            remaining_after_this = max(1, total - idx + 1)
            cap_this_chunk = min(20, max(0, round(remaining_budget / remaining_after_this)))
            if cap_this_chunk <= 0:
                return chunk_results

            # Stage 2: factextract
            # Only include abstract on first chunk to reduce repeated abstract-driven pairs.
            chunk_abstract = resolved_abstract if idx == 1 else ""
            facts = extract_atomic_facts_for_chunk(
                title=resolved_title,
                abstract=chunk_abstract,
                body=chunk_text,
                source_label=source_label,
                max_facts=max(20, cap_this_chunk * 4),
            )
            if not facts:
                if progress_callback:
                    progress_callback(f"Chunk {idx}: No atomic facts extracted")
                return chunk_results

            # Stage 3: generator_v2
            curriculum_phase = (idx - 1) % 6
            base_cap = max(1, cap_this_chunk // 3)
            base_pairs = generate_pairs_for_chunk(
                fact_list=facts,
                title=resolved_title,
                abstract=chunk_abstract,
                body=chunk_text,
                source_label=source_label,
                curriculum_phase=curriculum_phase,
                cap_this_chunk=base_cap,
                total_target=max_pairs,
                produced_so_far=current_produced,
                remaining_chunks=remaining_after_this - 1,
            )
            if not base_pairs:
                if progress_callback:
                    progress_callback(f"Chunk {idx}: No base pairs generated")
                return chunk_results

            # Stage 4: variation
            variations = generate_variations_for_pairs(
                base_pairs=base_pairs,
                title=resolved_title,
                abstract=chunk_abstract,
                body=chunk_text,
            )

            # Candidate pool sent to final review.
            candidate_pairs = base_pairs + variations
            if cap_this_chunk is not None and cap_this_chunk >= 0 and len(candidate_pairs) > cap_this_chunk:
                candidate_pairs = candidate_pairs[:cap_this_chunk]

            # Stage 5: reviewer_v2
            for pair in candidate_pairs:
                # Check if we've reached max pairs
                with lock:
                    if len(accepted_pairs) >= max_pairs:
                        return chunk_results
                
                # Check for duplicates (needs lock)
                with lock:
                    if is_dup_question(pair["question"], existing_questions):
                        continue
                
                # Review pair (or skip review for speed)
                if skip_review:
                    # Quick metadata check even when review is skipped (less aggressive)
                    q_lower = pair.get("question", "").lower()
                    a_lower = pair.get("answer", "").lower()
                    # Only filter obvious metadata, not normal words
                    metadata_keywords = ["file://", "path://", "http://", "https://", "metadata:", "e-mel:", "@", ".com"]
                    if any(keyword in q_lower or keyword in a_lower for keyword in metadata_keywords):
                        continue  # Skip pairs with obvious metadata
                    reviewed = dict(pair)
                else:
                    reviewed, _reason = review_pair(
                        pair,
                        chunk_text,
                        title=resolved_title,
                        abstract=chunk_abstract,
                    )
                
                if reviewed and isinstance(reviewed, dict):
                    reviewed["source"] = reviewed.get("source") or source_label
                    reviewed["chunk_text"] = chunk_text
                    if resolved_source and not reviewed.get("document_source"):
                        reviewed["document_source"] = resolved_source
                    # Add to results with lock
                    with lock:
                        # Check again if we've reached max
                        if len(accepted_pairs) >= max_pairs:
                            return chunk_results
                        # Double-check duplicates after lock
                        if not is_dup_question(reviewed["question"], existing_questions):
                            accepted_pairs.append(reviewed)
                            existing_questions.append(reviewed["question"])
                            chunk_results.append(reviewed)
        except InterruptedError:
            raise  # Allow cancellation to propagate
        except Exception as e:
            if progress_callback:
                progress_callback(f"Error in chunk {idx}: {str(e)}")

        return chunk_results
    
    # Process chunks in parallel using ThreadPoolExecutor
    chunk_data_list = [(chunk_text, source_name, idx, total_chunks) 
                       for idx, (chunk_text, _start, _end) in enumerate(chunks, 1)]
    
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all chunks
        future_to_chunk = {
            executor.submit(process_chunk, chunk_data): chunk_data[2] 
            for chunk_data in chunk_data_list
        }
        
        # Process completed chunks as they finish
        for future in as_completed(future_to_chunk):
            chunk_idx = future_to_chunk[future]
            completed += 1
            
            try:
                chunk_results = future.result()
                if progress_callback:
                    with lock:
                        current_count = len(accepted_pairs)
                    progress_callback(
                        f"Completed chunk {completed}/{total_chunks} | "
                        f"Total pairs: {current_count}/{max_pairs}"
                    )
            except InterruptedError:
                raise  # Allow cancellation to propagate
            except Exception as e:
                if progress_callback:
                    progress_callback(f"Error processing chunk {chunk_idx}: {str(e)}")
            
            # Check if we've reached max pairs (continue processing to check all chunks)
            with lock:
                if len(accepted_pairs) >= max_pairs:
                    # Note: We don't cancel futures as they may already be running
                    # The chunk processing function checks max_pairs internally
                    pass
    
    # Sort by source order for consistency
    accepted_pairs = sorted(accepted_pairs, key=lambda x: x.get('source', ''))
    
    if progress_callback:
        progress_callback(f"Completed! Generated {len(accepted_pairs)} Q&A pairs.")
    
    return accepted_pairs
