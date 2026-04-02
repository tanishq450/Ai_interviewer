import asyncio
import sys
from pathlib import Path

# Allow running this file directly: python Data/question_ingestor.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Data.question import QuestionEmbeddings
from qdrant.qdrant import QdrantHybridClient
from utils.Data_ingestion import Docloader


# PDFs expected next to project root (add files here or update the list)
PDF_PATHS = [
    PROJECT_ROOT / "AvenueE_Interview_Questions_updated7-27-2020_0.pdf",
    PROJECT_ROOT / "Top-100-Finance-Interview-Questions-Answers.pdf",
]


def detect_domain(text: str) -> str:
    text = text.lower()

    if any(w in text for w in ["sql", "database", "api", "python", "ml", "ai"]):
        return "tech"

    if any(w in text for w in ["finance", "valuation", "roi", "market"]):
        return "finance"

    if any(w in text for w in ["team", "conflict", "leadership"]):
        return "hr"

    return "general"


async def ingest_all():

    # Init
    qdrant = QdrantHybridClient()

    try:
        await qdrant.create_collection("question_collection")
    except Exception:
        pass

    embedder = QuestionEmbeddings(qdrant=qdrant)
    loader = Docloader()

    # -------- Load PDFs (skip missing; load_pdf returns None on failure) --------
    parts: list[str] = []
    for path in PDF_PATHS:
        path = Path(path)
        if not path.is_file():
            print(f"WARNING: PDF not found, skipping: {path}", flush=True)
            continue
        text = loader.load_pdf(str(path))
        if not text:
            print(f"WARNING: no text extracted from: {path}", flush=True)
            continue
        parts.append(text)

    all_text = "\n".join(parts)
    if not all_text.strip():
        raise SystemExit(
            "No PDF text loaded. Add PDFs next to the project root or fix paths in "
            "Data/question_ingestor.py (PDF_PATHS)."
        )

    lines = all_text.split("\n")

    questions = []

    for line in lines:
        line = line.strip()

        
        if "?" in line and len(line) > 10:
            questions.append({
                "question": line,
                "domain": detect_domain(line),   
                "topic": "general",
                "difficulty": "medium"
            })

    print(f"Extracted {len(questions)} questions")

    # -------- Ingest --------
    await embedder.ingest(questions)

    print("Ingestion complete")


if __name__ == "__main__":
    asyncio.run(ingest_all())

