# Agent-X

Deep dive into Generative AI with the 5 day Google workshop conducted on Kaggle.

- Prompt Engineering
- Embeddings and Similarity
- Retrieval Agumented Generation (RAG)
- Agents
- LangGraph
- Fine-Tuning
- Google Gemini API

## Prompt Engineering 
- Temperature, Top-K and Top-P Parameters
- Zero Shot
- One Shot
- Few Shot
- Chain of Thought
- Reason and Act (ReAct)

## Embeddings and Similarity
A similarity score of two embedding vectors can be obtained by calculating their inner product. If u is the first embedding vector, and v the second, this is u.v gives you the similarity score.  As these embedding vectors are normalised to unit length, this is also the cosine similarity.

This score can be computed across all embeddings through the matrix self-multiplication: df @ df.T.

Note that the range from 0.0 (completely dissimilar) to 1.0 (completely similar) is depicted in the heatmap from dark (0.0) to light (1.0).

```
# Generate embeddings
response = genai.embed_content(model='models/text-embedding-004', content=texts, task_type='semantic_similarity')

# Set up the embeddings in a dataframe.
df = pd.DataFrame(response['embedding'], index=texts)

# Perform the similarity calculation
similarity = df @ df.T
```

## Embeddings and Classification Task

Use the embeddings produced by the Gemini API to train a model that can classify newsgroup posts into the categories (the newsgroup itself) from the post contents.
This technique uses the Gemini API's embeddings as input, avoiding the need to train on text input directly, and as a result it is able to perform quite well using relatively few examples compared to training a text model from scratch.

```
# Generate embeddings for input text. You will be performing classification, so set task_type accordingly.
    response = genai.embed_content(
        model="models/text-embedding-004", content=text, task_type="classification"
    )
```

## Retrieval Agumented Generation (RAG)

Two big limitations of LLMs are 1) that they only "know" the information that they were trained on, and 2) that they have limited input context windows. A way to address both of these limitations is to use a technique called Retrieval Augmented Generation, or RAG. A RAG system has three stages:

- Indexing: Indexing happens ahead of time, and allows you to quickly look up relevant information at query-time.
- Retrieval:  When a query comes in, you retrieve relevant documents, combine them with your instructions and the user's query
- Generation: Have the LLM generate a tailored answer in natural language using the supplied information

This allows you to provide information that the model hasn't seen before, such as product-specific knowledge or live weather updates.

Use the Gemini API

- Create a vector database
- Retrieve answers to questions from the database
- Generate a final answer

```
passage_oneline = passage.replace("\n", " ")
query_oneline = query.replace("\n", " ")

# This prompt is where you can specify any guidance on tone, or what topics the model should stick to, or avoid.
prompt = f"""You are a helpful and informative bot that answers questions using text from the reference passage included below. 
Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. 
However, you are talking to a non-technical audience, so be sure to break down complicated concepts and 
strike a friendly and converstional tone. If the passage is irrelevant to the answer, you may ignore it.

QUESTION: {query_oneline}
PASSAGE: {passage_oneline}
"""

model = genai.GenerativeModel("gemini-1.5-flash-latest")
answer = model.generate_content(prompt)
Markdown(answer.text)
```

## Agents and Function Calling
Use the Gemini API's automatic function calling to build a chat interface over a local database. 

```
db_tools = [list_tables, describe_table, execute_query]

instruction = """You are a helpful chatbot that can interact with an SQL database for a computer
store. You will take the users questions and turn them into SQL queries using the tools
available. Once you have the information you need, you will answer the user's question using
the data returned. Use list_tables to see what tables are present, describe_table to understand
the schema, and execute_query to issue an SQL SELECT query."""

model = genai.GenerativeModel(
    "models/gemini-1.5-flash-latest", tools=db_tools, system_instruction=instruction
)
```

```
# Start a chat with automatic function calling enabled.
chat = model.start_chat(enable_automatic_function_calling=True)
```

```
resp = chat.send_message("What is the cheapest product?", request_options=retry_policy)
print(resp.text)
```

```
- DB CALL: list_tables
 - DB CALL: describe_table
 - DB CALL: execute_query
The cheapest product is the Mouse, costing $29.99.
```

## LangGraph
LangGraph applications are built around a **graph** structure. As the developer, you define an application graph that models the state transitions for your application. Your app will define a **state** schema, and an instance of that schema is propagated through the graph.

Each **node** in the graph represents an action or step that can be taken. Nodes will make changes to the state in some way through code that you define. These changes can be the result of invoking an LLM, by calling an API, or executing any logic that the node defines.

Each **edge** in the graph represents a transition between states, defining the flow of the program. Edge transitions can be fixed, for example if you define a text-only chatbot where output is always displayed to a user, you may always transition from chatbot -> user. The transitions can also be conditional, allowing you to add branching (like an if-else statement) or looping (like for or while loops).

LangGraph is highly extensible and provides a number of features such as memory, persistance and streaming.

```
# Set up the initial graph based on our state definition.
graph_builder = StateGraph(OrderState)

# Add the chatbot function to the app graph as a node called "chatbot".
graph_builder.add_node("chatbot", chatbot)

# Define the chatbot node as the app entrypoint.
graph_builder.add_edge(START, "chatbot")

chat_graph = graph_builder.compile()
```

## Fine-Tuning

## Google Gemini API

## Evolution of LLM Models
- GPT 1
	- Unsupervised PreTraining
	- Unlabelled Text

- GPT 2 
	- 10 times larger dataset for training, 10 times the parameters, larger brain, whole universe of text
	- generate super realistic texts, coherent, accuracy

- GPT 3
	- 175 billlion parameters
	- Few shot learning
	- no need for mountains of labelled data

- GPT 3.5 and 4
	- Doubled down on dialogue, multi modal 
	- Handle images and text as well - Multi Modal
	- longer context window
	- remember longer

- LamDA - Language Model for Dialogue Applications
	- smooth talker
	- natural sounding , human sounding, less robotic

- Gopher
	- refining the training process
	- data quality and optimization

- PalM Pathways Language Model
	- 540 billion parameters
	- Common sense reasoning
	- Arithmetic Reasoning
	
- PalM-2
	- code generation
	- works with fewer parameters
	- Smarter, better reasoning, code generation

- Gemini
	- multi modal 
	- text, video, audio
	- Swiss Army knife
	- millions of tokens

- Gemini Nano
	- AI to smaller devices like Smartphones and Wearables

- Meta
	- Llama

- Mistral AI
	- MistralLM

- xAI
	- Grok1

## Fine Tuning Techniques
	- Supervised finetuneing SFT
		- become expert in a specific area
		- give a crash course
	- RLHF - reinforcement learning with human feedback
		- Reward Model
			- good vs bad model — talent show
	- PEFT Parameter Efficient Fine Tuning
			- retrain specific parts
			- Adapters 


## Sampling Techniques

## Inference Optimization Techniques
		- Quantization
			- Lower Precision
		- Distillation	
			- Train a smaller, faster student model
		- Output preserving methods
			- Flash Attention
		- Prefix Caching
			- Chatbots
			- Long Documents
			- Cache Previous outputs - remember previous answers
		- Speculative Decoding
			- Team of Assistants working in parallel

