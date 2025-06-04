# Onboarding Agent

## 1) Onboarding Agent Idea
The company we work for handles a high volume of daily interactions between the support team and application users. Customers ask a wide range of questions — from app installation to product customization, including banners, colors, margins, sales conditions, and more.
The purpose of this agent is to reduce the manual workload for our support team. Since we offer a wide variety of customizations, our manuals are updated frequently. Every time we release a new feature, fix a bug, simplify a customization process, or remove an unused feature, the manual must be updated accordingly. This requires close collaboration between the support and development teams. The support staff must stay informed about all changes, fully understand them and be able to use and explain the new features to the customers.
With nearly 50 applications and only about 60 customer support staff, it is easy to see how overwhelming it can be for any one person to keep track of 5–10 applications, monitor ongoing changes, and confidently explain every feature.
Our goal is to help the customer support team by reducing the number of incoming tickets. This would give them more time to deepen their knowledge of the applications and work more effectively with the development team.

## 2) About the Onboarding Agent
The idea behind our agent is straightforward. Each time the manual is updated, the latest version is uploaded to the application. The old manual data is removed from the database, and the new content is vectorized and stored.

## 3) How to use the application
When the application is started (see description below), it can be accessed via link provided in the terminal and it is http://192.168.0.19:5001/. There is nothing that customers or the support team need to do. The customers can start prompting the agent and the agent will be responding immediately.

## 4) Setup
a) Create the virtual environment (from the project root):
```bash
python3.12 -m venv .venv
```
b) Activate the virtual environment:
```bash
source .venv/bin/activate
```
c) Deactivate the virtual environment (when you're done):
```bash
deactivate
```

## 5) Run the project
a) Start the application by running (from the project root):
```bash
python main.py 
```
(Wait until the debugger is activated and the server is ready.)
b) Open your browser and navigate to:
```
http://192.168.0.19:5001/
```
c) Enter prompts in the interface — the agent will respond with answers.

## 6) Evaluate agent
a) Run the evaluation script:
```bash
python model_eval.py
```
b) Click on the link that is generated in the terminal.
c) Or open it in your browser to view the evaluation dashboard.
d) Review the results to analyze the agent's performance.

## 7) Technologies
This application leverages an AI-powered architecture built primarily using the LangChain framework to enable dynamic document-based question answering. The system integrates several technologies and services. It uses Groq’s LLMs (e.g., LLaMA3-8B-8192) for natural language generation, alongside MistralAI’s embedding models for semantic search and vector similarity operations. The manuals are vectorized using these embeddings and stored in an in-memory vector database for quick retrieval. 
To manage and structure the flow of logic, the system utilizes LangGraph’s StateGraph to define a multi-node conversational workflow that processes inputs, determines whether to trigger tool usage (e.g., document retrieval), and then creates responses based on contextual information. 
Document ingestion is handled through a DOCX parser (via python-docx), and documents are split into chunks using LangChain’s RecursiveCharacterTextSplitter. The system ensures that each manual update reflects the most current application state. The database is cleared every time the application is started and the data is replaced with updated embeddings.
For secure environment configuration, the application retrieves API keys (LangSmith, Groq, Hugging Face, and Mistral) from a secured Google Drive.

Package dependencies are validated and installed at runtime, ensuring the system is equipped with libraries such as langchain_community, beautifulsoup4, and pandas.
All interactions and updates are traced via LangSmith for observability. The inclusion of LangGraph's memory checkpointing feature (MemorySaver) enables persistent thread tracking and debugging — similar in nature to version control systems like Git — making this application maintainable.

## 8) Agent Evaluation Overview
In order to ensure the accuracy of the agent’s final responses, we leverage LangSmith’s evaluation framework. A dataset of sample question–answer pairs is created to represent typical user interactions with the onboarding agent. Each example is evaluated using an LLM-as-a-judge methodology.
The grading model—llama3-70b-8192 accessed via Groq—is configured for structured output and follows strict evaluation criteria. It receives the user question, the ground truth answer, and the agent’s response. The model then determines whether the response is factually accurate, allowing for additional information as long as it does not conflict with the expected answer.
The evaluation process is fully asynchronous and runs through LangSmith’s aevaluate() method with built-in concurrency support. Once completed, LangSmith provides a link to the evaluation results, which can also be exported as a DataFrame for further analysis.

## 9) Frontend

## 10) Next steps
As this agent is intended for customer support, it should have the ability to not automatically respond to customer requests. The application should support two modes: Automatic and Manual.
In Automatic mode, the agent responds to customer queries immediately.
In Manual mode, a sound notification alerts the support team of a new request, and the customer is informed that the first available agent will respond shortly. This mode allows the support team to review and edit responses before sending them.
Additionally, the application should support automatic translation of responses based on the customer’s location. This feature should be customizable:
In automatic mode, translations occur without intervention.
In manual mode, the support team selects the target language before responding.
All conversations between customers and the support team will be stored in a separate knowledge database. This data will be used to complement the main manual during the next scheduled update. If the app owner does not perform the update within 7 days, a background process will automatically trigger it, incorporating the new knowledge from support team–customer interactions into the main manual.
Rather than relying on an in-memory vector database, the application should use a persistent (external) vector database for storing and retrieving embeddings.

TODO
Provide keys via Google Drive
