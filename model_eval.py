# Response evauluator
#https://docs.smith.langchain.com/evaluation/concepts#evaluating-an-agents-final-response
from typing import Literal
from main import graph
import json
import asyncio
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.types import Command, interrupt
from typing_extensions import Annotated, TypedDict
from langsmith import Client

client = Client()

# Create a dataset
examples = [
    {
        "inputs": {
            "question": "What is a Google Merchant?",
        },
        "outputs": {
            "response": "Google Merchant Center is a centralized dashboard"
            "that allows you to organize your e-commerce products to appear"
            "in Google shopping searches, allowing potential customers to browse your products via Google."
        }
    },
    {
        "inputs": {
            "question": "Do I need to have a Google account?",
        },
        "outputs": {
            "response": "A standard Google account is a prerequisite."
            "If you have one, verify that you are logged in. If not, you can set one up."
        }
    },
    {
        "inputs": {
            "question": "Do I need to provide my personal data?",
        },
        "outputs": {
            "response": "Google will ask you to input details, including your name and contact information."
            "You'll also need to indicate whether your business is online, brick-and-mortar, or both."
        }
    }
]

dataset_name = "SC Onboarding Agent"

if not client.has_dataset(dataset_name=dataset_name):
    dataset = client.create_dataset(dataset_name=dataset_name)
    client.create_examples(
        dataset_id=dataset.id,
        examples=examples
    )


# LLM-as-judge instructions
grader_instructions = """You are a teacher grading a quiz.

You will be given a QUESTION, the GROUND TRUTH (correct) RESPONSE, and the STUDENT RESPONSE.

Here is the grade criteria to follow:
(1) Grade the student responses based ONLY on their factual accuracy relative to the ground truth answer.
(2) Ensure that the student response does not contain any conflicting statements.
(3) It is OK if the student response contains more information than the ground truth response, as long as it is factually accurate relative to the  ground truth response.

Correctness:
True means that the student's response meets all of the criteria.
False means that the student's response does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct."""


# LLM-as-judge output schema
class Grade(TypedDict):
    """Compare the expected and actual answers and grade the actual answer."""
    reasoning: Annotated[str, ..., "Explain your reasoning for whether the actual response is correct or not."]
    is_correct: Annotated[bool, ..., "True if the student response is mostly or exactly correct, otherwise False."]


# Judge LLM
grader_llm = init_chat_model("llama3-70b-8192", temperature=0, model_provider="groq").with_structured_output(Grade, method="json_mode")


# Evaluator function
async def final_answer_correct(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """Evaluate if the final response is equivalent to reference response."""
    user = f"""QUESTION: {inputs['question']}
    GROUND TRUTH RESPONSE: {reference_outputs['response']}
    STUDENT RESPONSE: {outputs['response']}"""
    grade = await grader_llm.ainvoke([{"role": "system", "content": grader_instructions}, {"role": "user", "content": user}])
    return grade["is_correct"]


# Response eval
async def run_graph(inputs: dict) -> dict:
    result = await graph.ainvoke({
        "messages":
        [{ "role": "user", "content": inputs['question']}]},
        config = {"env": "test", "configurable": {"thread_id": "test_123"}})
    return {"response": result}


async def evaluate_model():
    return await client.aevaluate(
    run_graph,
    data=dataset_name,
    evaluators=[final_answer_correct],
    experiment_prefix="SC-Onboarding-Agent",
    num_repetitions=1,
    max_concurrency=4,
)


results = asyncio.run(evaluate_model())
results.to_pandas()