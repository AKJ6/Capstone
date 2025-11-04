import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import csv
import re

try:
    import requests
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'requests'. Install with: pip install requests"
    ) from exc

# Optional: scikit-learn for TF-IDF retrieval. We'll fall back gracefully if missing.
try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False


OLLAMA_HOST_DEFAULT = "http://localhost:11434"
OLLAMA_CHAT_ENDPOINT = "/api/chat"
DEFAULT_MODEL = os.environ.get("LLAMA_MODEL", "llama3.1")


def build_system_prompt() -> str:
    return (
        "You are a world-class culinary assistant and recipe developer. "
        "Generate clear, complete, and executable recipes. "
        "Always tailor the recipe to the provided ingredients, dietary preferences, and cuisine. "
        "If something is missing, make reasonable assumptions and state them. "
        "Use metric units with US conversions in parentheses where helpful. "
        "Return well-structured sections: Title, Description, Ingredients (with amounts), Equipment, "
        "Instructions (numbered steps), Timing, Serving Size, Variations/Substitutions, "
        "Allergens, and Approximate Nutrition per serving."
    )


def build_user_prompt(
    ingredients: List[str],
    diet: Optional[str],
    cuisine: Optional[str],
    servings: Optional[int],
    max_calories: Optional[int],
    extra_tools: Optional[List[str]],
    retrieved_context: Optional[str] = None,
) -> str:
    parts: List[str] = []
    if retrieved_context:
        parts.append("Reference context (retrieved snippets):\n" + retrieved_context.strip())
        parts.append("")
    parts.append("Please create a complete recipe with the following constraints:")
    parts.append("")
    parts.append(f"- Ingredients on hand: {', '.join(ingredients) if ingredients else 'user did not specify'}")
    parts.append(f"- Dietary preference: {diet or 'none specified'}")
    parts.append(f"- Cuisine style: {cuisine or 'chef\'s choice, but be consistent'}")
    if servings:
        parts.append(f"- Target servings: {servings}")
    if max_calories:
        parts.append(f"- Aim for ≤ {max_calories} kcal per serving")
    if extra_tools:
        parts.append(f"- Available equipment: {', '.join(extra_tools)}")
    parts.append("")
    parts.append(
        "Constraints and style:"\
        "\n- Prefer fresh, seasonal choices when possible"\
        "\n- Avoid rare/expensive ingredients unless necessary"\
        "\n- Provide substitutions if key items are missing"\
        "\n- Include timing estimates per step and total time"\
        "\n- Include food safety notes if applicable"
    )
    parts.append("")
    parts.append(
        "Output format (use clear headings):\n"
        "Title\nDescription\nIngredients\nEquipment\nInstructions\nTiming\nServing Size\n"
        "Variations and Substitutions\nAllergens\nApproximate Nutrition per serving"
    )
    return "\n".join(parts)


# ----------------------------- RAG Utilities -----------------------------

def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _load_corpus(corpus_path: str) -> List[Dict[str, str]]:
    """Load a corpus file. Supports CSV or TXT.

    CSV heuristic: contains headers; we will try common recipe fields.
    Returns a list of dicts with keys: 'text' and 'meta'.
    """
    if not os.path.exists(corpus_path):
        raise SystemExit(f"Corpus file not found: {corpus_path}")

    _, ext = os.path.splitext(corpus_path.lower())
    documents: List[Dict[str, str]] = []

    if ext in {".csv"}:
        with open(corpus_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Try to build a useful text blob. Fall back to joining all fields.
                possible_fields = [
                    row.get("translatedRecipeName") or row.get("name") or row.get("title"),
                    row.get("ingredients_list") or row.get("ingredients") or row.get("TranslatedIngredients") or row.get("cleaned_ingredients"),
                    row.get("translatedRecipeInstructions") or row.get("instructions") or row.get("method"),
                    row.get("cuisine") or row.get("region") or row.get("course"),
                ]
                fields_used = [v for v in possible_fields if v]
                if not fields_used:
                    # join any non-empty fields
                    fields_used = [str(v) for v in row.values() if v]
                combined = " | ".join(fields_used)
                text_blob = combined.strip()
                if not text_blob:
                    continue
                meta = row.get("translatedRecipeName") or row.get("name") or row.get("title") or "entry"
                documents.append({"text": text_blob, "meta": str(meta)})
    else:
        # Treat as plain text; split by blank lines
        with open(corpus_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        chunks = [c.strip() for c in re.split(r"\n\s*\n", content) if c.strip()]
        for i, chunk in enumerate(chunks):
            documents.append({"text": chunk, "meta": f"chunk_{i+1}"})

    if not documents:
        raise SystemExit("Corpus appears to be empty after loading.")
    return documents


def _retrieve_context(
    query_text: str,
    documents: List[Dict[str, str]],
    top_k: int = 3,
    max_context_chars: int = 1200,
) -> Tuple[str, List[Tuple[str, float]]]:
    """Retrieve top_k relevant snippets.

    Returns a tuple: (context_block, [(meta, score), ...])
    """
    query = _normalize_text(query_text)
    docs_text = [d["text"] for d in documents]

    scores: List[float] = []
    if _SKLEARN_AVAILABLE:
        try:
            vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
            matrix = vectorizer.fit_transform([_normalize_text(t) for t in docs_text + [query]])
            doc_matrix = matrix[:-1]
            q_vec = matrix[-1]
            sims = cosine_similarity(doc_matrix, q_vec).ravel()
            scores = sims.tolist()
        except Exception:
            # Fallback to Jaccard if vectorizer blows up
            _sk_fallback = True
            scores = []
            query_tokens = set(query.split())
            for t in docs_text:
                tokens = set(_normalize_text(t).split())
                inter = len(tokens & query_tokens)
                union = len(tokens | query_tokens) or 1
                scores.append(inter / union)
    else:
        # Simple Jaccard similarity
        query_tokens = set(query.split())
        for t in docs_text:
            tokens = set(_normalize_text(t).split())
            inter = len(tokens & query_tokens)
            union = len(tokens | query_tokens) or 1
            scores.append(inter / union)

    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: max(1, top_k)]
    snippets: List[str] = []
    meta_scores: List[Tuple[str, float]] = []
    total_chars = 0
    for i in ranked_idx:
        snippet = documents[i]["text"].strip()
        meta = documents[i]["meta"]
        score = float(scores[i])
        if not snippet:
            continue
        # Truncate each snippet to keep within max_context_chars total
        remaining = max_context_chars - total_chars
        if remaining <= 0:
            break
        snippet_trimmed = snippet[: max(200, min(remaining, 600))]
        snippets.append(f"[Source: {meta}]\n{snippet_trimmed}")
        meta_scores.append((meta, score))
        total_chars += len(snippet_trimmed)

    context_block = "\n\n".join(snippets)
    return context_block, meta_scores


def call_ollama_chat(
    model: str,
    system_prompt: str,
    user_prompt: str,
    host: str = OLLAMA_HOST_DEFAULT,
    temperature: float = 0.7,
    top_p: float = 0.95,
    seed: Optional[int] = None,
) -> str:
    url = host.rstrip("/") + OLLAMA_CHAT_ENDPOINT
    headers = {"Content-Type": "application/json"}
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": temperature, "top_p": top_p},
    }
    if seed is not None:
        payload["options"]["seed"] = seed

    try:
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=240)
    except requests.exceptions.ConnectionError as err:
        raise SystemExit(
            "Could not connect to Ollama at {}. Ensure Ollama is installed, running, and the model '{}' is pulled (e.g., 'ollama run {}').".format(
                host, model, model
            )
        ) from err
    except requests.exceptions.Timeout as err:
        raise SystemExit("Request to Llama (Ollama) timed out. Try again or adjust inputs.") from err

    if resp.status_code != 200:
        raise SystemExit(
            f"Ollama returned HTTP {resp.status_code}: {resp.text[:500]}"
        )

    data = resp.json()
    # Ollama chat returns { 'message': { 'content': '...' }, ... }
    message = data.get("message", {})
    content = message.get("content")
    if not content:
        # Some older/newer variants may return 'response'
        content = data.get("response")
    if not content:
        raise SystemExit("No content returned by model.")
    return content


def write_output(text: str, out_path: Optional[str] = None) -> str:
    if not out_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(os.getcwd(), f"generated_recipe_llama_{timestamp}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_path


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a recipe using a local Llama model via Ollama."
    )
    parser.add_argument(
        "--ingredients",
        type=str,
        default="",
        help="Comma-separated list of ingredients on hand",
    )
    parser.add_argument(
        "--diet",
        type=str,
        default="",
        help="Dietary preference (e.g., vegetarian, vegan, gluten-free)",
    )
    parser.add_argument(
        "--cuisine",
        type=str,
        default="",
        help="Cuisine style (e.g., Indian, Italian, Mexican)",
    )
    parser.add_argument(
        "--servings",
        type=int,
        default=None,
        help="Target number of servings",
    )
    parser.add_argument(
        "--max_calories",
        type=int,
        default=None,
        help="Aim for this many kcal per serving (approximate)",
    )
    parser.add_argument(
        "--equipment",
        type=str,
        default="",
        help="Comma-separated list of available equipment/tools",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Ollama model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("OLLAMA_HOST", OLLAMA_HOST_DEFAULT),
        help=f"Ollama server host (default: {OLLAMA_HOST_DEFAULT})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Nucleus sampling top_p",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic output (if supported)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Path to save the generated recipe (defaults to timestamped .txt)",
    )
    parser.add_argument(
        "--rag",
        action="store_true",
        help="Enable retrieval-augmented generation (RAG) using a local corpus",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="",
        help="Path to corpus file (CSV or TXT) to use for RAG",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of retrieved snippets to include",
    )
    parser.add_argument(
        "--max_context_chars",
        type=int,
        default=1200,
        help="Maximum total characters of retrieved context",
    )

    args = parser.parse_args(argv)
    return args


def interactive_fallback(args: argparse.Namespace) -> None:
    if not args.ingredients:
        args.ingredients = input("Enter ingredients (comma-separated): ").strip()
    if not args.diet:
        args.diet = input("Enter dietary preference (or leave blank): ").strip()
    if not args.cuisine:
        args.cuisine = input("Enter cuisine (or leave blank): ").strip()
    if args.servings is None:
        try:
            sv = input("Target servings (leave blank to skip): ").strip()
            args.servings = int(sv) if sv else None
        except ValueError:
            args.servings = None
    if args.max_calories is None:
        try:
            kc = input("Max kcal per serving (leave blank to skip): ").strip()
            args.max_calories = int(kc) if kc else None
        except ValueError:
            args.max_calories = None
    if not args.equipment:
        args.equipment = input("Available equipment/tools (comma-separated, optional): ").strip()


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # If the three core inputs are not provided, prompt interactively
    if not args.ingredients or not args.diet or not args.cuisine:
        print("One or more required fields missing. Enter details interactively.")
        interactive_fallback(args)

    ingredients_list = [s.strip() for s in args.ingredients.split(",") if s.strip()]
    equipment_list = [s.strip() for s in args.equipment.split(",") if s.strip()]

    # Build a simple query string for retrieval
    rag_context: Optional[str] = None
    rag_sources: List[Tuple[str, float]] = []
    if args.rag:
        if not args.corpus:
            print("RAG enabled but no --corpus provided; skipping retrieval.")
        else:
            try:
                documents = _load_corpus(args.corpus)
                query_pieces = [
                    "ingredients: " + ", ".join(ingredients_list) if ingredients_list else "",
                    f"diet: {args.diet}" if args.diet else "",
                    f"cuisine: {args.cuisine}" if args.cuisine else "",
                ]
                query_text = " | ".join([p for p in query_pieces if p]) or "general cooking recipe"
                rag_context, rag_sources = _retrieve_context(
                    query_text=query_text,
                    documents=documents,
                    top_k=max(1, int(args.top_k or 3)),
                    max_context_chars=max(300, int(args.max_context_chars or 1200)),
                )
            except SystemExit as e:
                # Bubble up human-friendly errors
                raise
            except Exception as e:
                print(f"RAG retrieval failed: {e}. Continuing without RAG.")
                rag_context = None

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(
        ingredients=ingredients_list,
        diet=args.diet or None,
        cuisine=args.cuisine or None,
        servings=args.servings,
        max_calories=args.max_calories,
        extra_tools=equipment_list if equipment_list else None,
        retrieved_context=rag_context,
    )

    print(f"\nGenerating recipe with model '{args.model}' via {args.host}...\n")
    content = call_ollama_chat(
        model=args.model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        host=args.host,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )

    print(content)

    if rag_sources:
        # Print a brief reference list with scores
        print("\nReferences (retrieved):")
        for meta, score in rag_sources:
            print(f"- {meta} (score={score:.3f})")

    out_path = write_output(content, args.out or None)
    if rag_sources:
        try:
            refs_path = os.path.splitext(out_path)[0] + "_refs.txt"
            with open(refs_path, "w", encoding="utf-8") as rf:
                rf.write("References (retrieved)\n")
                for meta, score in rag_sources:
                    rf.write(f"- {meta} (score={score:.3f})\n")
            print(f"Saved references to: {refs_path}")
        except Exception:
            pass
    print(f"\nSaved to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())



