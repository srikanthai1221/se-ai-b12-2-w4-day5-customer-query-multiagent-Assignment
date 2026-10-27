from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from rag_tool import search_knowledge_base

load_dotenv()
llm = LLM(model="gpt-4o-mini", temperature=0.2)

kb_agent = Agent(
    role="Knowledge Base Specialist",
    goal="Answer customer questions accurately using only the ShopEase knowledge base",
    backstory=(
        "You work in ShopEase customer support. You always search the knowledge base "
        "before answering and never invent policy details."
    ),
    tools=[search_knowledge_base],
    llm=llm,
    verbose=True,
)

kb_task = Task(
    description=(
        "Customer question: {question}\n"
        "Search the knowledge base and answer the question. "
        "If the knowledge base does not contain the answer, reply exactly: NOT_FOUND_IN_KB"
    ),
    expected_output="A short, accurate answer based on the knowledge base, or NOT_FOUND_IN_KB.",
    agent=kb_agent,
)

if __name__ == "__main__":
    crew = Crew(agents=[kb_agent], tasks=[kb_task], verbose=True)
    result = crew.kickoff(inputs={"question": "How many days do I have to return a product?"})
    print("\nFINAL ANSWER:\n", result)