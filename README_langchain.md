# Learning Generative AI

The objective is to learn how develop Generative AI app using the *langchain* / *langgraph* library framework.

## Runnable

- Runnable interface (common methods)
    - invoke()
    - stream()
    - \_\_or__()
    - \_\_ror__()

- Concrete Runnables
    - RunnableLambda
    - RunnableSequence
    - RunnablePassthrough
    - RunnableParallel
    - RunnableEach
    - RunnableBinding
    - RunnableLike
    - RunnableAssign
    - RunnablePick
    - RunnableBranch
    - RunnableWithFallbacks
    - RunnableRetry
    - RouterRunnable
    - RunnableConfigurableFields
    - RunnableConfigurableAlternatives

These Runnables are wrappers around other objects to implement additional logics and provide a common Runnable interface. Some of these Runnables are created using these methods:

Runnable
- assign() -> chain with RunnableBinding
- bind() -> RunnableBinding
- with_config() -> RunnableBinding
- with_listeners() -> RunnableBinding
- with_alisteners() -> RunnableBinding
- with_types() -> RunnableBinding
- with_retry() -> RunnableRetry
- map() -> RunnableEach
- with_fallbacks() -> RunnableWithFallbacks

RunnableSerializable(Runnable)
- configurable_fields -> RunnableConfigurableFields
- configurable_alternatives -> RunnableConfigurableAlternatives

## Chaining

- Chaining of Runnables
    - Chaining of Runnables return a new Runnable
    - Chaining is implemented by Python "or" (|) operator
    - Invoking the chain will invoke all the Runnables in the chain
    - Chain can be composed into serial or parallel branches to form a tree-like structure of Runnables
    - Input and output data from each Runnable are passed along the chain
    
- Objects that implemented Runnable
    - Prompt
    - LLM Model
    - Chat Model
    - Output Parser
    - Tool

Prompt, Chat Model, and Output Parser can be chained together to allow input and output data of these components to flow through the chain.

## Generative AI Model

LangChain provide these base classes for other vendor-specific implementation

- BaseLanguageModel
- BaseChatModel(BaseLanguageModel)
- SimpleChatModel(BaseChatModel)
- BaseLLM(BaseLanguageModel)
- LLM(BaseLLM)
- Embeddings

### Vendor-specific Langchain Libraries

#### Google

Python library **langchain-google-genai**

- GoogleGenerativeAI(BaseLLM)
- ChatGoogleGenerativeAI(BaseChatModel)
- GoogleGenerativeAIEmbeddings(Embeddings)

#### Azure

Python library **langchain-azure-ai**

- AzureAIChatCompletionsModel(BaseChatModel)

#### OpenAI

Python library **langchain-openai**

- BaseChatOpenAI(BaseChatModel)
- ChatOpenAI(BaseChatOpenAI)
- BaseOpenAI(BaseLLM)
- OpenAI(BaseOpenAI)
- AzureOpenAI(BaseOpenAI)
- AzureChatOpenAI(BaseChatOpenAI)

The native Python library is **openai**

#### Ollama

Python library **langchain-ollama**

- ChatOllama(BaseChatModel)
- OllamaLLM(BaseLLM)
- OllamaEmbeddings(Embeddings)

### GenAI Model Configuration

GenAI model has many attributes to be configured. Many of these attributes can only be configured when the model object is created. After creation, these attributes cannot simply be updated. However, there are methods that can duplicate the object with some attributes modified.

Runnable
- bind()
- with_config()

These methods allow override per invoke(config: RunnableConfig) via "configurable" entry 

RunnableSerializable
- configurable_fields()
- configurable_alternatives()

Typical attributes for a GenAI model are:
- model
- temperature
- top_p
- top_k
- max_output_tokens
- n
- max_retries

Each vendor will have its own additional attributes that can be configured on the model.

## Prompt