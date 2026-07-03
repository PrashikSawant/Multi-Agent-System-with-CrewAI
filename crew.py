"""
crew.py
Day 25 - Multi-Agent System with CrewAI

Defines a 3-agent crew:
1. Researcher  — searches and gathers facts on a topic
2. Analyst     — takes the research, finds key insights
3. Writer      — turns the analysis into a clean, readable report

Each agent has a role, goal, and backstory — these get injected into
that agent's system prompt automatically by CrewAI. The output of each
task becomes the input for the next one (sequential process).
"""

# Monkey-patch fix for CrewAI bug #5886:
# CrewAI injects cache_breakpoint into all messages regardless of provider.
# Groq rejects this field with a 400 error. This patch disables the injection.
import crewai.llms.cache as _crewai_cache
_crewai_cache.mark_cache_breakpoint = lambda msg: msg

import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool

# ---- LLM configuration ----
# CrewAI 1.x uses LiteLLM under the hood, which needs the provider prefix
# "groq/" before the model name so LiteLLM knows which API to route to.
groq_llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3,
)

# ---- Search tool ----
# SerperDev is a paid search API with a free tier (2,500 searches/month)
# More reliable than ddgs for CrewAI's tool integration
search_tool = SerperDevTool()


# ---- Agents ----
researcher = Agent(
    role="Senior Research Analyst",
    goal=(
        "Find accurate, current, and relevant information about {topic}. "
        "Focus on facts, recent developments, and credible sources."
    ),
    backstory=(
        "You are a meticulous research analyst with 10 years of experience "
        "gathering and verifying information across technology, business, and "
        "science. You only report what you can verify, and always note your sources."
    ),
    tools=[search_tool],
    llm=groq_llm,
    verbose=True,
    max_iter=3,
)

analyst = Agent(
    role="Strategic Insights Analyst",
    goal=(
        "Analyze research findings about {topic} and extract the 3-5 most "
        "important insights. Identify patterns, implications, and what this "
        "means in practical terms."
    ),
    backstory=(
        "You are a strategic analyst who turns raw research into actionable "
        "insights. You cut through noise and identify what actually matters, "
        "explaining the 'so what' behind every finding."
    ),
    llm=groq_llm,
    verbose=True,
)

writer = Agent(
    role="Technical Content Writer",
    goal=(
        "Write a clear, engaging, well-structured report about {topic} based "
        "on the analysis provided. Make it readable for a technical audience "
        "without unnecessary jargon."
    ),
    backstory=(
        "You are a technical writer who specializes in making complex topics "
        "accessible. You write with clarity and structure, using headers, "
        "bullet points where appropriate, and a professional but engaging tone."
    ),
    llm=groq_llm,
    verbose=True,
)


# ---- Tasks ----
research_task = Task(
    description=(
        "Research the topic: {topic}. "
        "Search for recent developments, key facts, and important context. "
        "Gather at least 5 relevant facts or data points. "
        "Note the source for each fact."
    ),
    expected_output=(
        "A structured list of research findings about {topic} - "
        "at least 5 facts/data points, each with a source noted."
    ),
    agent=researcher,
)

analysis_task = Task(
    description=(
        "Analyze the research findings provided and extract the 3-5 most "
        "important insights about {topic}. "
        "For each insight, explain: what it is, why it matters, and what "
        "the practical implication is."
    ),
    expected_output=(
        "A structured analysis with 3-5 key insights about {topic}, "
        "each with a clear explanation of significance and practical implications."
    ),
    agent=analyst,
    context=[research_task],
)

writing_task = Task(
    description=(
        "Write a professional research report about {topic} based on the "
        "analysis provided. "
        "Structure it with: an executive summary, key findings section, "
        "implications, and a brief conclusion. "
        "Aim for 400-600 words, clear and engaging."
    ),
    expected_output=(
        "A well-structured research report (400-600 words) about {topic} "
        "with executive summary, key findings, implications, and conclusion."
    ),
    agent=writer,
    context=[analysis_task],
)


# ---- Crew ----
research_crew = Crew(
    agents=[researcher, analyst, writer],
    tasks=[research_task, analysis_task, writing_task],
    process=Process.sequential,
    verbose=True,
)


def run_crew(topic: str) -> str:
    """Kicks off the full crew and returns the final written report."""
    result = research_crew.kickoff(inputs={"topic": topic})
    return str(result)