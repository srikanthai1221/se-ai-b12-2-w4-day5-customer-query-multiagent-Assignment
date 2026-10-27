from datetime import date

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool

load_dotenv()
llm = LLM(model="gpt-4o-mini", temperature=0.2)

# Ready-made Google search tool (reads SERPER_API_KEY from .env)
search_tool = SerperDevTool(n_results=5)

web_agent = Agent(
    role="Web Research Specialist",
    goal="Find accurate, up-to-date public information on the web to answer customer questions",
    backstory=(
        "You support the ShopEase customer care team. When a question is about public "
        "information (products, regulations, general facts), you search the web, use reliable "
        "sources, and never make up facts. You do NOT answer questions about ShopEase's own "
        "policies from the web."
    ),
    tools=[search_tool],
    llm=llm,
    verbose=True,
)

web_task = Task(
        description=(
        f"Today's date is {date.today()}.\n"
        "Customer question: {question}\n"
        "Search the web and give a short, factual answer. "
        "Mention the source website(s) you used."
    ),
    expected_output="A short factual answer (3-5 sentences) with the source website names or links.",
    agent=web_agent,
)

if __name__ == "__main__":
    crew = Crew(agents=[web_agent], tasks=[web_task], verbose=True)
    result = crew.kickoff(inputs={"question": "What is the current per-transaction limit for UPI payments in India?"})
    print("\nFINAL ANSWER:\n", result)