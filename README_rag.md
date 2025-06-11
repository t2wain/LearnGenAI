# Retrieved-Augmented Generation (RAG)

The components of RAG are:

- Text data
- Embedding model
- VectorStore
- Retriever

RAG system provide data as context in LLM prompt to reduce hallucination effect.

## Document

Data object in LangChain

- BaseMedia(Serializable)
    - id: Optional[str]
    - metadata: dict
- Blob(BaseMedia)
    - data: Union[bytes, str, None]
    - mimetype: Optional[str]
    - encoding: str = "utf-8"
    - path: Optional[PathLike]
    - from_path(cls, path: PathLike) -> Blob:
    - from_data(cls, data: Union[str, bytes]) -> Blob:
- Document(BaseMedia)
    - page_content: str

## Document Loaders

Common methods of loading document using LangChain

- BaseLoader(ABC)
    - lazy_load(self) -> Iterator[Document]
    - load(self) -> list[Document]
    - load_and_split(self, text_splitter: Optional[TextSplitter] = None) -> list[Document]
- BaseBlobParser(ABC)
    - lazy_parse(self, blob: Blob) -> Iterator[Document]
    - parse(self, blob: Blob) -> list[Document]

#### Unstructured library

- UnstructuredLoader(BaseLoader)
- UnstructuredBaseLoader(BaseLoader, ABC) - deprecated
- UnstructuredFileLoader(UnstructuredBaseLoader) - deprecated

## PDF  Loaders

Parsing PDF file into text

- BasePDFLoader(BaseLoader, ABC)
- UnstructuredPDFLoader(UnstructuredFileLoader)
- OnlinePDFLoader(BasePDFLoader)
- PyPDFLoader(BasePDFLoader)
- PyPDFium2Loader(BasePDFLoader)
- PyPDFDirectoryLoader(BaseLoader)
- PDFMinerLoader(BasePDFLoader)
- PyMuPDFLoader(BasePDFLoader)
- PDFPlumberLoader(BasePDFLoader)
- DedocPDFLoader(DedocBaseLoader)

## Embedding model

The embedding model returns a vector of decimal number (embedding) for an input document (or prompt). The embedding and the document are stored in a vector store. Two embeddings, such as from a prompt and a document, can be compared mathematically for similarity. Vector store can retrieve documents based on prompt query by calculating similarity matches between embeddings. 

- Embeddings
    - embed_documents(self, texts: list[str]) -> list[list[float]]
    - embed_query(self, text: str) -> list[float]
- GoogleGenerativeAIEmbeddings(BaseModel, Embeddings)
- OllamaEmbeddings(BaseModel, Embeddings)
- OpenAIEmbeddings(BaseModel, Embeddings)
- AzureOpenAIEmbeddings(OpenAIEmbeddings)

## Vector Store and Retriever

Vector store provides implementation for generating embeddings (with a given embedding model) and storing both embeddings and documents. Retriever provides implementation for querying documents from a vector store. Typical implementation combines both of these functionalities.

- BaseRetriever(RunnableSerializable[RetrieverInput, RetrieverOutput], ABC)
    - invoke(self, input: str, config: Optional[RunnableConfig] = None, **kwargs: Any) -> list[Document]
- VectorStoreRetriever(BaseRetriever)
- VectorStore(ABC)
    - embeddings(self) -> Optional[Embeddings]
    - as_retriever(self, **kwargs: Any) -> VectorStoreRetriever

### Implementations of vector store

- InMemoryVectorStore(VectorStore)
    - \_\_init__(self, embedding: Embeddings) -> None
- Chroma(VectorStore)
- FAISS(VectorStore)

## Text Splitter

Large document should be split into smaller documents before adding to the vector store. Only relevant portion of a document should be included as context in a prompt to minimize the size of the prompt.

- TextSplitter(BaseDocumentTransformer, ABC)
- RecursiveCharacterTextSplitter(TextSplitter)
