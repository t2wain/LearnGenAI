# Learning Generative AI

The objective is to learn how develop Generative AI app using the *langchain* / *langgraph* library framework.

## Runnable

Runnable interface (common methods)
- invoke()
- stream()
- \_\_or__()
- \_\_ror__()

Concrete Runnables

- RunnableSerializable(Serializable, Runnable)
- RunnableLambda(Runnable)
- RunnableSequence(RunnableSerializable)
- RunnablePassthrough(RunnableSerializable)
- RunnableParallel(RunnableSerializable)
- RunnableEachBase(RunnableSerializable)
- RunnableEach(RunnableEachBase)
- RunnableBindingBase(RunnableSerializable)
- RunnableBinding(RunnableBindingBase)
- RunnableWithMessageHistory(RunnableBindingBase)

- RunnableLike - typing
- coerce_to_runnable(thing: RunnableLike) -> Runnable
- chain() -> RunnableLambda (Decorate a function to make it a Runnable)

- RunnablePassthrough(RunnableSerializable)
- RunnableAssign(RunnableSerializable)
- RunnablePick(RunnableSerializable)
- RunnableBranch(RunnableSerializable)
- RunnableWithFallbacks(RunnableSerializable)
- RunnableRetry(RunnableSerializable)
- RouterRunnable(RunnableSerializable)
- DynamicRunnable(RunnableSerializable)
- RunnableConfigurableFields(DynamicRunnable)
- RunnableConfigurableAlternatives(DynamicRunnable)

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

Chaining of Runnables
- Chaining of Runnables return a new Runnable
- Chaining is implemented by Python "or" (|) operator
- Invoking the chain will invoke all the Runnables in the chain
- Chain can be composed into serial or parallel branches to form a tree-like structure of Runnables
- Input and output data from each Runnable are passed along the chain
    
Objects that implemented Runnable
- Prompt
- LLM Model
- Chat Model
- Output Parser
- Tool

Prompt, Chat Model, and Output Parser can be chained together to allow input and output data of these components to flow through the chain.

## Generative AI Model

LangChain provide these base classes that will be inherited by vendor-specific implementations to provide a common API

```python
LanguageModelInput = Union[PromptValue, str, Sequence[MessageLikeRepresentation]]
LanguageModelOutput = Union[BaseMessage, str]
LanguageModelLike = Runnable[LanguageModelInput, LanguageModelOutput]
LanguageModelOutputVar = TypeVar("LanguageModelOutputVar", BaseMessage, str)
```

- BaseLanguageModel(RunnableSerializable[LanguageModelInput, LanguageModelOutputVar])
    - get_num_tokens(self, text: str) -> int
    - get_num_tokens_from_messages(self, messages: list[BaseMessage], tools: Optional[Sequence] = None) -> int
- BaseChatModel(BaseLanguageModel)
    - invoke(self, input: LanguageModelInput) -> BaseMessage
    - stream(self, input: LanguageModelInput) -> Iterator[BaseMessageChunk]:
- SimpleChatModel(BaseChatModel)
- BaseLLM(BaseLanguageModel)
    - invoke(self, input: LanguageModelInput) -> str
    - stream(self, input: LanguageModelInput) -> Iterator[str]
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
    - save(self, file_path: Union[Path, str]) -> None
    - partial(self, **kwargs: Union[str, Callable[[], str]]) -> BasePromptTemplate
    - format(self, **kwargs: Any) -> FormatOutputType
    - invoke(input: dict) -> PromptValue
    - format_prompt(self, **kwargs: Any) -> PromptValue
- StringPromptTemplate(BasePromptTemplate, ABC)
- PromptTemplate(StringPromptTemplate)
- FewShotPromptWithTemplates(StringPromptTemplate)
- ImagePromptTemplate(BasePromptTemplate)

Chat model

- BaseChatPromptTemplate(BasePromptTemplate, ABC)
    - format_messages(self, **kwargs: Any) -> list[BaseMessage]
    - format_prompt(self, **kwargs: Any) -> PromptValue
    - pretty_repr(self, html: bool = False) -> str
    - pretty_print(self) -> None
- ChatPromptTemplate(BaseChatPromptTemplate)
    - from_template(cls, template: str, **kwargs: Any) -> ChatPromptTemplate
    - from_messages(cls, messages: Sequence[MessageLikeRepresentation]) -> ChatPromptTemplate:
- FewShotChatMessagePromptTemplate(BaseChatPromptTemplate)

Prompt for chat model consists of messages. These are messages that allows input variables.

- BaseMessagePromptTemplate(ABC)
    - format_messages(self, **kwargs: Any) -> list[BaseMessage]
    - pretty_repr(self, html: bool = False) -> str
    - pretty_print(self) -> None
    - \_\_add__(self, other: Any) -> ChatPromptTemplate
- BaseStringMessagePromptTemplate(BaseMessagePromptTemplate, ABC)
- MessagesPlaceholder(BaseMessagePromptTemplate)
- _StringImageMessagePromptTemplate(BaseMessagePromptTemplate)
    - prompt: Union[StringPromptTemplate, list[Union[StringPromptTemplate, ImagePromptTemplate]]]
    - from_template(cls, template: Union[str, list[Union[str, _TextTemplateParam, _ImageTemplateParam]]]) -> Self
    - from_template_file(cls, template_file: Union[str, Path], input_variables: list[str]) -> Self
    - format(self, **kwargs: Any) -> BaseMessage
- HumanMessagePromptTemplate(_StringImageMessagePromptTemplate)
- AIMessagePromptTemplate(_StringImageMessagePromptTemplate)
- SystemMessagePromptTemplate(_StringImageMessagePromptTemplate)
- ChatMessagePromptTemplate(BaseStringMessagePromptTemplate)

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

- BasePromptTemplate(ABC)
    - invoke(input: dict) -> PromptValue
    - format_prompt(self, **kwargs: Any) -> PromptValue

- PromptValue(ABC)
- StringPromptValue(PromptValue)
    - text: str
- ChatPromptValue(PromptValue)
    - messages: Sequence[BaseMessage]
- ImagePromptValue(PromptValue)
    - image_url: ImageURL(TypedDict)

## Messages

Messages are input and out unit values when invoking chat model.

- BaseMessage(Serializable)
    - type: str
- ChatMessage(BaseMessage)
    - role: str
    - type: Literal["chat"] = "chat"
- AIMessage(BaseMessage)
    - type: Literal["ai"] = "ai"
- HumanMessage(BaseMessage)
    - type: Literal["human"] = "human"
- SystemMessage(BaseMessage)
    - type: Literal["system"] = "system"
- FunctionMessage(BaseMessage)
    - type: Literal["function"] = "function"
- RemoveMessage(BaseMessage)
    - type: Literal["remove"] = "remove"

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

- BaseMessage(Serializable)
    - content: Union[str, list[Union[str, dict]]]

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

A response message (*AIMessage*) may be a tool call instruction.

- AIMessage(BaseMessage)
    - tool_calls: list[ToolCall] = []
    - invalid_tool_calls: list[InvalidToolCall] = []

Parsers

- ToolCall(TypedDict)
- InvalidToolCall(TypedDict)

- tool_call() -> ToolCall
- invalid_tool_call() -> InvalidToolCall
- default_tool_parser() -> tuple[list[ToolCall], list[InvalidToolCall]]

 The result of a tool call is a tool message. Tool message can be added to the chat prompt to provide a complete sequence of communication.

- ToolMessage(BaseMessage)

#### AIMessage metadata (usage_metadata)

- AIMessage(BaseMessage)
    - usage_metadata: Optional[UsageMetadata] = None

- UsageMetadata(TypedDict)
- InputTokenDetails(TypedDict)
- OutputTokenDetails(TypedDict)

- add_usage()
- subtract_usage()

## Output Parser

Model can be instructed to output structured response such as csv, list, xml, or tool calls. Langchain provides various parser to return a Python data object.

- BaseOutputParser(BaseLLMOutputParser, RunnableSerializable[LanguageModelOutput, T])
    - invoke(self, input: Union[str, BaseMessage]) -> T
- JsonOutputParser
    - get_format_instructions(self) -> str
- ListOutputParser
- CommaSeparatedListOutputParser
    - get_format_instructions(self) -> str
- NumberedListOutputParser
    - get_format_instructions(self) -> str
- MarkdownListOutputParser
    - get_format_instructions(self) -> str
- StrOutputParser
- XMLOutputParser
    - get_format_instructions(self) -> str

OpenAI functions

- OutputFunctionsParser
    - parse_result(self, result: list[Generation], *, partial: bool = False) -> Any
- JsonOutputFunctionsParser
- JsonKeyOutputFunctionsParser
- PydanticOutputFunctionsParser
    - pydantic_schema: Union[type[BaseModel], dict[str, type[BaseModel]]]
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
    - pydantic_object: Annotated[type[TBaseModel], SkipValidation()]

## Structured Output

Model can be configured to respond with structured output. Each vendor has its own specific implementation. Typical implementation returns a chain of a model with additional field binding (tools) and a corresponding output parser.

BaseChatModel
- with_structured_output() -> Runnable

```python
def with_structured_output(
        self,
        schema: Union[typing.Dict, type],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable
```

**schema**:

The output schema. Can be passed in as:

- an OpenAI function/tool schema,
- a JSON Schema,
- a TypedDict class,
- or a Pydantic class.

If ``schema`` is a Pydantic class then the *model output* will be a *Pydantic instance* of that class, and the model-generated fields *will be validated by the Pydantic class*. Otherwise the model output *will be a dict and will not be validated*. See `langchain_core.utils.function_calling.convert_to_openai_tool` for more on how to properly specify types and descriptions of schema fields when specifying a Pydantic or TypedDict class.

**include_raw**:

If False then only the parsed structured output is returned. If an error occurs during model output parsing it will be raised. If True then both the raw model response (a BaseMessage) and the parsed model response will be returned. If an error occurs during output parsing it will be caught and returned as well. The final output is always a dict with keys "raw", "parsed", and "parsing_error".

Returns:

A Runnable that takes same inputs as a `langchain_core.language_models.chat.BaseChatModel`.

If ``include_raw`` is False and ``schema`` is a Pydantic class, Runnable outputs an instance of ``schema`` (i.e., a Pydantic object). 

Otherwise, if ``include_raw`` is False then Runnable outputs a dict.

If ``include_raw`` is True, then Runnable outputs a dict with keys:

- ``"raw"``: BaseMessage
- ``"parsed"``: None if there was a parsing error, otherwise the type depends on the ``schema`` as described above.
- ``"parsing_error"``: Optional[BaseException]

Example: Pydantic schema (include_raw=False):

```python
from pydantic import BaseModel

class AnswerWithJustification(BaseModel):
    '''An answer to the user question along with justification for the answer.'''
    answer: str
    justification: str

llm = ChatModel(model="model-name", temperature=0)
structured_llm = llm.with_structured_output(AnswerWithJustification)

structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
```
```python
AnswerWithJustification(
     answer='They weigh the same',
     justification=(
        'Both a pound of bricks and a pound of feathers weigh one pound. '
        'The weight is the same, but the volume or density of the objects may differ.'
     )
)
```

Example: Pydantic schema (include_raw=True):

```python
from pydantic import BaseModel

class AnswerWithJustification(BaseModel):
    '''An answer to the user question along with justification for the answer.'''
    answer: str
    justification: str

llm = ChatModel(model="model-name", temperature=0)
structured_llm = llm.with_structured_output(AnswerWithJustification, include_raw=True)

structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
```
```python
{
    'raw': AIMessage(
        content='', 
        additional_kwargs={
            'tool_calls': [
                {'id': 'call_Ao02pnFYXD6GN1yzc0uXPsvF', 
                'function': {
                    'arguments': (
                        '{"answer":"They weigh the same.","justification":"Both a pound of bricks '
                        'and a pound of feathers weigh one pound. The weight is the same, '
                        'but the volume or density of the objects may differ."}'
                    ), 
                    'name': 'AnswerWithJustification'}, 
                    'type': 'function'
                }
            ]
        }
    ),
    'parsed': AnswerWithJustification(
        answer='They weigh the same.', 
        justification=(
            'Both a pound of bricks and a pound of feathers weigh one pound. '
            'The weight is the same, but the volume or density of the objects may differ.'
        )
    ),
    'parsing_error': None
}
```

Example: Dict schema (include_raw=False):

```python
from pydantic import BaseModel
from langchain_core.utils.function_calling import convert_to_openai_tool

class AnswerWithJustification(BaseModel):
    '''An answer to the user question along with justification for the answer.'''
    answer: str
    justification: str

dict_schema = convert_to_openai_tool(AnswerWithJustification)
llm = ChatModel(model="model-name", temperature=0)
structured_llm = llm.with_structured_output(dict_schema)

structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
```
```python
{
    'answer': 'They weigh the same',
    'justification': (
        'Both a pound of bricks and a pound of feathers weigh one pound. '
        'The weight is the same, but the volume and density of the two substances differ.'
    )
}
```

#### Ollama

```python
def with_structured_output(
        self,
        schema: Union[dict, type],
        *,
        method: Literal["function_calling", "json_mode", "json_schema"] = "json_schema",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[dict, BaseModel]]
```

## Tools

BaseChatModel
- bind_tools() -> Runnable

```python
def bind_tools(
        self,
        tools: Sequence[
            Union[typing.Dict[str, Any], type, Callable, BaseTool]
        ],
        *,
        tool_choice: Optional[Union[str]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
```

These langchain objects can be used to define "tool". Each vendor will has its own specific implementation of bind_tools(). Typically, it will call convert_to_openai_function() or convert_to_openai_tool() to generate schema to configure the underlying AI model.

- BaseTool(RunnableSerializable)
- Tool(BaseTool)
- StructuredTool(BaseTool)
- ToolException(Exception)

- create_schema_from_function() -> BaseModel
- get_all_basemodel_annotations() -> dict[str, type]
- tool() -> Union[BaseTool, Callable[[Union[Callable, Runnable]], BaseTool]] (decorator)
- convert_runnable_to_tool() -> BaseTool
- render_text_description(tools: list[BaseTool]) -> str
- render_text_description_and_args(tools: list[BaseTool]) -> str

- RetrieverInput(BaseModel)
- create_retriever_tool() -> Tool

- FunctionDescription(TypedDict)
- ToolDescription(TypedDict)
- convert_to_openai_function() -> dict[str, Any]
- convert_to_openai_tool() -> dict[str, Any]
- convert_to_json_schema() -> dict[str, Any]
- tool_example_to_messages() -> list[BaseMessage]

## Pydantic

Many langchain object is a Pydantic object:

- BaseModel
- Serializable(BaseModel, ABC)
- RunnableSerializable(Serializable, Runnable)
- BaseMessage(Serializable)
- BaseTool(RunnableSerializable)

Using pydantic for the following features:

- deserialize to json
- provide typing information to facilitate parsing and validation
- provide description for object attributes that can be used within prompt

Use pydantic when define the following:

- Tool
- Structured output