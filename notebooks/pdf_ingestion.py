"""Ingestion et découpage de fichiers PDF pour un pipeline RAG LCEL."""

from pathlib import Path
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFIngestion:
    """Charge un PDF, nettoie son texte et le découpe en chunks."""

    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 150):
        if chunk_size <= 0:
            raise ValueError("chunk_size doit être strictement positif.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap doit être positif et inférieur à chunk_size."
            )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )

    @staticmethod
    def _clean(text: str) -> str:
        """Normalise les espaces sans supprimer le contenu extrait."""
        text = re.sub(r"[ \t\f\v]+", " ", text)
        text = re.sub(r"\s*\n\s*", "\n", text)
        return text.strip()

    def process(self, path: str | Path):
        """Charge un .pdf et renvoie ses chunks sous forme de Documents LangChain."""
        path = Path(path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Document PDF introuvable : {path}")
        if path.suffix.casefold() != ".pdf":
            raise ValueError(f"Le fichier doit être au format .pdf : {path}")

        documents = PyPDFLoader(str(path)).load()
        for document in documents:
            document.page_content = self._clean(document.page_content)
            # PyPDFLoader ajoute déjà le numéro de page aux métadonnées.
            document.metadata["source"] = path.name

        return self.splitter.split_documents(documents)
