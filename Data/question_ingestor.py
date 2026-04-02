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


path1 = "/home/tanishq/AI_interviewer/AvenueE_Interview_Questions_updated7-27-2020_0.pdf"
path2 = "/home/tanishq/AI_interviewer/Top-100-Finance-Interview-Questions-Answers.pdf"


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

    # -------- Load PDFs --------
    text1 = loader.load_pdf(path1)
    text2 = loader.load_pdf(path2)

    all_text = text1 + "\n" + text2


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

