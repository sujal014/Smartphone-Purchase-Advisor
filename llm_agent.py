"""
LangChain component for the Smartphone Purchase Advisor.

Does two pieces of real language work (not a single throwaway call):

1. extract_requirements()
   Takes the user's free-text query (e.g. "I need a phone under 30k with a
   great camera, I don't game much but battery matters a lot") and uses an
   LLM with structured output to pull out a budget and 1-10 importance
   weights for camera / performance / battery. This is genuine extraction:
   the model has to interpret vague language ("don't game much" -> low
   performance importance; "battery matters a lot" -> high battery
   importance) and fill in sensible defaults for anything unstated.

2. explain_recommendation()
   Takes the fuzzy-logic ranking (numbers only) and turns it into a
   natural, conversational explanation of *why* the top phone fits and
   what the trade-offs of the alternatives are.
"""

import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Groq model to use. "llama-3.3-70b-versatile" is a strong free-tier model
# that supports structured output / tool calling well.
# GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

class UserRequirements(BaseModel):
    budget: int = Field(
        description="Maximum budget in INR the user is willing to spend. "
        "If a range is given, use the upper bound. Default to 25000 if not mentioned."
    )
    camera_importance: int = Field(
        ge=1, le=10,
        description="How important camera quality is to the user, 1 (not important) "
        "to 10 (very important). Infer from context if not explicit; default 5.",
    )
    performance_importance: int = Field(
        ge=1, le=10,
        description="How important performance/gaming/speed is to the user, 1-10. "
        "Infer from context (e.g. 'I game a lot' => high); default 5.",
    )
    battery_importance: int = Field(
        ge=1, le=10,
        description="How important battery life is to the user, 1-10. Infer from "
        "context (e.g. 'battery dies fast on my current phone' => high); default 5.",
    )
    summary: str = Field(
        description="One short sentence summarizing what the user is looking for."
    )


def _get_llm(temperature: float = 0):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it as an environment variable "
            "(locally) or as a Streamlit secret (when deployed)."
        )
    return ChatGroq(model=GROQ_MODEL, temperature=temperature, api_key=api_key)


def extract_requirements(user_query: str) -> UserRequirements:
    """Uses the LLM to turn a free-text request into structured requirements."""
    llm = _get_llm()
    structured_llm = llm.with_structured_output(UserRequirements)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a smartphone shopping assistant. Read the user's request "
                "carefully and extract their budget and how important camera, "
                "performance, and battery life are to them, on a 1-10 scale. "
                "Use context clues and common sense to infer unstated values "
                "rather than defaulting everything to 5 — e.g. someone who "
                "mentions gaming cares about performance; someone who mentions "
                "photography or Instagram cares about the camera; someone who "
                "travels a lot or complains about charging cares about battery.",
            ),
            ("human", "{query}"),
        ]
    )
    chain = prompt | structured_llm
    return chain.invoke({"query": user_query})


def explain_recommendation(user_query: str, requirements: UserRequirements, ranked_phones: list) -> str:
    """Uses the LLM to turn the fuzzy-logic ranking into a conversational
    explanation, referencing the actual reasons for the ranking."""
    llm = _get_llm(temperature=0.4)

    phones_text = "\n".join(
        f"{i+1}. {p['name']} — Rs.{p['price']:,} — fuzzy suitability score: {p['score']}/100"
        for i, p in enumerate(ranked_phones[:5])
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a friendly, concise smartphone shopping advisor. You are "
                "given the user's original request, their extracted requirements, "
                "and a ranked list of phones with suitability scores produced by a "
                "fuzzy logic engine. Write a short (4-6 sentence) conversational "
                "recommendation: name the top pick and explain in plain language "
                "why it suits them, then briefly mention one alternative and what "
                "trade-off it involves. Do not mention 'fuzzy logic' or the raw "
                "scores explicitly — speak naturally, like a helpful salesperson.",
            ),
            (
                "human",
                "User request: {query}\n\n"
                "Extracted requirements: budget=Rs.{budget}, camera importance="
                "{camera_importance}/10, performance importance={performance_importance}/10, "
                "battery importance={battery_importance}/10\n\n"
                "Ranked phones:\n{phones}",
            ),
        ]
    )
    chain = prompt | llm
    response = chain.invoke(
        {
            "query": user_query,
            "budget": requirements.budget,
            "camera_importance": requirements.camera_importance,
            "performance_importance": requirements.performance_importance,
            "battery_importance": requirements.battery_importance,
            "phones": phones_text,
        }
    )
    return response.content
