from time import sleep
from functools import lru_cache

import openai
from langchain_core.prompts import PromptTemplate
from src.retriever import retrieve_docs
from src.config import (
    FALLBACK_MODEL_PROVIDER,
    HUGGINGFACE_MODEL_ID,
    MODEL_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)

# Hugging Face imports
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# OpenAI imports
from langchain_openai import ChatOpenAI


prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful travel assistant.
Use the following travel guide context to answer the question.
If the answer is not found, say you don't know — don’t make it up.

Context:
{context}

Question:
{question}

Answer:"""
)


@lru_cache(maxsize=1)
def _get_openai_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model_name=OPENAI_MODEL,
        temperature=0.7,
        max_retries=3,
        timeout=30,
    )


@lru_cache(maxsize=1)
def _get_huggingface_model():
    tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(HUGGINGFACE_MODEL_ID)
    return tokenizer, model


def _build_prompt(context: str, question: str) -> str:
    return prompt_template.format(context=context, question=question)


def _invoke_with_retries(payload: dict[str, str]):
    retry_delays = (1, 2, 4)
    qa_chain = prompt_template | _get_openai_llm()

    for attempt, delay in enumerate(retry_delays, start=1):
        try:
            return qa_chain.invoke(payload)
        except (openai.InternalServerError, openai.APIConnectionError, openai.APITimeoutError):
            if attempt == len(retry_delays):
                raise RuntimeError("The language model service is temporarily unavailable. Please try again in a moment.")
            sleep(delay)
        except openai.RateLimitError:
            raise RuntimeError("The language model is rate-limited right now. Please retry shortly.")
        except openai.OpenAIError as error:
            raise RuntimeError(f"OpenAI request failed: {error}") from error


def _generate_with_huggingface(context: str, question: str) -> str:
    tokenizer, model = _get_huggingface_model()
    prompt = _build_prompt(context, question)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    output_tokens = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )
    return tokenizer.decode(output_tokens[0], skip_special_tokens=True).strip()

def generate_answer(query: str) -> str:
    docs = retrieve_docs(query)
    context = "\n".join(docs)

    if MODEL_PROVIDER == "openai":
        try:
            result = _invoke_with_retries({"context": context, "question": query})
            return result.content
        except RuntimeError:
            if FALLBACK_MODEL_PROVIDER == "huggingface":
                return _generate_with_huggingface(context, query)
            raise

    if MODEL_PROVIDER == "huggingface":
        return _generate_with_huggingface(context, query)

    raise ValueError(f"Unknown MODEL_PROVIDER: {MODEL_PROVIDER}")
