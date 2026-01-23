from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def build_knowledge_base(pdf_file):
    reader = PdfReader(pdf_file)
    text = "".join(page.extract_text() for page in reader.pages)

    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings()
    return Chroma.from_texts(chunks, embeddings)
