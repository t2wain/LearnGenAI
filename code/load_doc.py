from typing import Iterable
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from langchain.text_splitter import TextSplitter, RecursiveCharacterTextSplitter


def _get_pdf_parser(file_path: str) -> BaseLoader:
    return PyPDFLoader(
        file_path = file_path,
        # headers = None
        # password = None,
        mode = "single",
        pages_delimiter = "\n\f",
        # extract_images = True,
        # images_parser = RapidOCRBlobParser(),
    )


def load_pdf(file_path: str) -> list[Document]:
    return [doc for doc in _get_pdf_parser(file_path).lazy_load()]


def _get_text_splitter() -> TextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=1000, # Adjust chunk size as needed
        chunk_overlap=200 # Optional overlap for context preservation
    )


def split_doc(docs: Iterable[Document]) -> list[Document]:
    return _get_text_splitter().split_documents(docs)


def load_split_pdf(file_path: str) -> list[Document]:
    return split_doc(load_pdf(file_path))