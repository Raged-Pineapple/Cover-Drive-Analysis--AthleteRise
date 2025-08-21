import os
import json
import time
import google.generativeai as genai

# Configure lazily; do not raise at import time
_API_KEY = os.getenv("GEMINI_API_KEY")
if _API_KEY:
    try:
        genai.configure(api_key=_API_KEY)
    except Exception as _e:
        # Log and continue; function will handle fallback
        print(f"[ai_helper] Gemini configure warning: {_e}")

_DEF_TIPS = [
    "Keep head still and over the ball.",
    "Stride decisively into the line.",
    "Lead with top hand; elbow up.",
    "Bat face straight; play under the eyes.",
    "Transfer weight forward through contact.",
]

def _extract_text(resp) -> str:
    """Robustly extract text from a Gemini response even when .text is empty."""
    try:
        # Quick accessor
        if getattr(resp, "text", None):
            return resp.text
        # Candidates/parts path
        chunks = []
        for cand in getattr(resp, "candidates", []) or []:
            # Log finish_reason if present to aid debugging
            fr = getattr(cand, "finish_reason", None)
            if fr is not None:
                print(f"[ai_helper] candidate.finish_reason={fr}")
            content = getattr(cand, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []) or []:
                t = getattr(part, "text", None)
                if t:
                    chunks.append(t)
        text = "\n".join(chunks).strip()
        return text
    except Exception as e:
        print(f"[ai_helper] _extract_text error: {e}")
        return ""


def _select_models() -> list[str]:
    """Determine model preference order from env with sensible fallbacks."""
    env_model = os.getenv("GEMINI_MODEL", "").strip()
    chain = []
    if env_model:
        chain.append(env_model)
    # Default chain from newest → widely available
    chain.extend([
        "gemini-2.5-pro",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
    ])
    # Deduplicate preserving order
    seen = set()
    ordered = []
    for m in chain:
        if m and m not in seen:
            seen.add(m)
            ordered.append(m)
    return ordered

def generate_ai_recommendations(evaluation: dict) -> list[str]:
    """
    Generate 3–5 concise coaching tips using Gemini.
    Always returns a non-empty list with actionable tips.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        # No key available; return high-quality defaults
        return _DEF_TIPS[:5]

    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        print(f"[ai_helper] Gemini configure error: {e}")
        return _DEF_TIPS[:5]

    prompt = (
        "You are a professional cricket batting coach. Analyze this player's cover drive.\n"
        "Data (JSON):\n"
        f"{json.dumps(evaluation, indent=2)}\n\n"
        "Provide 3–5 short, actionable recommendations to improve technique. "
        "Use bullet points. Keep each under 15 words."
    )

    models = _select_models()
    # Retry parameters
    max_attempts_per_model = 3
    for model_name in models:
        try:
            model = genai.GenerativeModel(model_name)
        except Exception as e:
            print(f"[ai_helper] Unable to init model '{model_name}': {e}")
            continue
        for attempt in range(1, max_attempts_per_model + 1):
            try:
                resp = model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.4,
                        "max_output_tokens": 256,
                    },
                )
                text = _extract_text(resp)
                if not text:
                    # Log and retry with backoff
                    print(f"[ai_helper] Empty response from '{model_name}' (attempt {attempt}). Retrying...")
                    if attempt < max_attempts_per_model:
                        time.sleep(0.7 * attempt)
                        continue
                    else:
                        break
                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                tips = []
                for ln in lines:
                    ln = ln.lstrip("-•*0123456789. ").strip()
                    if ln:
                        tips.append(ln)
                # Deduplicate and clamp
                seen, deduped = set(), []
                for t in tips:
                    if t not in seen:
                        seen.add(t)
                        deduped.append(t)
                if deduped:
                    return deduped[:5]
                # If parsing yielded nothing, try next attempt
                print(f"[ai_helper] Parsed no tips from '{model_name}' (attempt {attempt}).")
                if attempt < max_attempts_per_model:
                    time.sleep(0.7 * attempt)
                
            except Exception as e:
                # Likely transient (5xx) or quota; retry then fallback model
                print(f"[ai_helper] Gemini error with '{model_name}' (attempt {attempt}): {e}")
                if attempt < max_attempts_per_model:
                    time.sleep(0.7 * attempt)
                    continue
                else:
                    break

    # All attempts failed → high-quality defaults
    return _DEF_TIPS[:5]
