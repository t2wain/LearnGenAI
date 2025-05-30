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

LangChain provide these base classes that will be inherited by vendor-specific implementations to provide a common API

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
- max_retries

Each vendor will have its own additional attributes that can be configured on their model.

## Prompt

Prompt is the input data when invoking the model. Prompt allows input variables to provide customization for each invocation.

Completion model

- BasePromptTemplate(ABC)
- StringPromptTemplate(BasePromptTemplate, ABC)
- PromptTemplate(StringPromptTemplate)
- FewShotPromptWithTemplates(StringPromptTemplate)
- ImagePromptTemplate(BasePromptTemplate)

Chat model

- BaseChatPromptTemplate(BasePromptTemplate, ABC)
- ChatPromptTemplate(BaseChatPromptTemplate)

Prompt for chat model consists of messages. These are messages that allows input variables.

- BaseMessagePromptTemplate(ABC)
- BaseStringMessagePromptTemplate(BaseMessagePromptTemplate, ABC)
- MessagesPlaceholder(BaseMessagePromptTemplate)
- _StringImageMessagePromptTemplate(BaseMessagePromptTemplate)
- HumanMessagePromptTemplate(_StringImageMessagePromptTemplate)
- AIMessagePromptTemplate(_StringImageMessagePromptTemplate)
- SystemMessagePromptTemplate(_StringImageMessagePromptTemplate)
- ChatMessagePromptTemplate(BaseStringMessagePromptTemplate)
- FewShotChatMessagePromptTemplate(BaseChatPromptTemplate)

Messages can be of the following types:

```python
MessageLike = Union[BaseMessagePromptTemplate, BaseMessage, BaseChatPromptTemplate]

MessageLikeRepresentation = Union[
    MessageLike,
    tuple[
        Union[str, type],
        Union[str, list[dict], list[object]],
    ],
    str,
    dict,
]
```

#### Prompt Value

A Prompt is a Runnable and when invoked return a PromptValue

- PromptValue(ABC)
- StringPromptValue(PromptValue)
- ChatPromptValue(PromptValue)
- ImagePromptValue(PromptValue)
    - image_url: ImageURL(TypedDict)

## Messages

Messages are input and out unit values when invoking chat model.

- BaseMessage
- ChatMessage(BaseMessage)
- AIMessage(BaseMessage)
- HumanMessage(BaseMessage)
- SystemMessage(BaseMessage)
- FunctionMessage(BaseMessage)
- RemoveMessage(BaseMessage)

Convenience methods to manipulate messages:

- get_buffer_string()
- messages_from_dict()
- convert_to_messages()
- messages_from_dict()
- filter_messages()
- merge_message_runs()
- trim_messages()
- convert_to_openai_messages()
- count_tokens_approximately()

#### Content Block

A message consists of one or more content blocks:

```python
content: Union[str, list[Union[str, dict]]]
```

- BaseDataContentBlock(TypedDict)
- URLContentBlock(BaseDataContentBlock)
- Base64ContentBlock(BaseDataContentBlock)
- PlainTextContentBlock(BaseDataContentBlock)
- IDContentBlock(TypedDict)

Convenience methods to manipulate content blocks:

- convert_to_openai_image_block()
- is_data_content_block()
- merge_content()

#### Tool calling

A response message (AIMessage) may be a tool call instruction. The result of a tool call is a tool message. Tool message can be added to the chat prompt to provide a complete sequence of communication.

- ToolMessage(BaseMessage)
- ToolCall(TypedDict)
- InvalidToolCall(TypedDict)

- tool_call() -> ToolCall
- invalid_tool_call() -> InvalidToolCall
- default_tool_parser() -> tuple[list[ToolCall], list[InvalidToolCall]]

A tool is a python function with input parameters and output object. Type annotation of input parameter and output object and help string will be used as further instruction to the model.

When pydantic object is used for input parameters and output object, the descriptive metadata of those pydantic object will be used as further instruction to the model.

#### AIMessage metadata (usage_metadata)

- UsageMetadata(TypedDict)
- InputTokenDetails(TypedDict)
- OutputTokenDetails(TypedDict)

- add_usage()
- subtract_usage()

## Output Parser

Model can be instructed to output structured response such as csv, list, xml, or tool calls. Langchain provides various parser to return a Python data object.

- JsonOutputParser
- ListOutputParser
- CommaSeparatedListOutputParser
- NumberedListOutputParser
- MarkdownListOutputParser
- StrOutputParser
- XMLOutputParser

OpenAI functions

- OutputFunctionsParser
- JsonOutputFunctionsParser
- JsonKeyOutputFunctionsParser
- PydanticOutputFunctionsParser
- PydanticAttrOutputFunctionsParser

OpenAI tools

- parse_tool_call()
- make_invalid_tool_call()
- parse_tool_calls()

- JsonOutputToolsParser
- JsonOutputKeyToolsParser
- PydanticToolsParser

Pydantic

- PydanticOutputParser

#### Structured Output

Model can be configured to respond with structured output in combination with instruction within the prompt sent to the model:

BaseChatModel
- bind_tools()
- with_structured_output()

#### Pydantic

Use pydantic library to define object with descriptive attributes. The descriptive metadata provides more instruction to the model so that the structured response from the model can be parsed into the same pydantic object.