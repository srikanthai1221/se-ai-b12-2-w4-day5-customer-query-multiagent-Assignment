from datetime import date
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM

from agent1_rag import kb_agent
from agent2_web import web_agent
from agent3_local import local_agent

load_dotenv()

# The manager: no tools, it only delegates to the 3 specialists
manager = Agent(
    role="Customer Support Manager",
    goal="Make sure every ShopEase customer gets a correct, friendly and complete answer",
    backstory=(
        "You lead the ShopEase support team. Read the customer's message and delegate to the right specialist:\n"
        "- Knowledge Base Specialist: ShopEase policies (returns, shipping, cancellation, payments, warranty, support hours)\n"
        "- Order Records Specialist: anything about a specific order or order ID\n"
        "- Web Research Specialist: public or general information that is not about ShopEase\n"
        "One message may need more than one specialist. Never answer from your own knowledge."
        "Never answer from your own knowledge. "
        "When you use 'Delegate work to coworker', the task and context must each be a "
        "plain text sentence, and coworker must be the exact role name of the specialist."
    ),
    llm=LLM(model="gpt-4o", temperature=0),
    allow_delegation=True,    
    max_iter=8,
    verbose=True,
)

support_task = Task(
    description=(
        f"Today's date is {date.today()}.\n"
        "Customer message: {question}\n\n"
        "Delegate to the right specialist(s), then write the final reply to the customer. "
        "If the customer asks about an order but gives no order ID, politely ask for it. "
        "If no specialist can answer, apologise and share ShopEase's support contact details "
        "(get them from the Knowledge Base Specialist)."
    ),
    expected_output="A friendly, accurate reply in 3-6 sentences, written directly to the customer.",
)

crew = Crew(
    agents=[kb_agent, web_agent, local_agent],  # the specialists
    tasks=[support_task],
    process=Process.hierarchical,               # manager decides who does what
    manager_agent=manager,
    verbose=True,
)

if __name__ == "__main__":
    print("ShopEase Customer Support (type 'exit' to quit)")
    while True:
        question = input("\nCustomer: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        result = crew.kickoff(inputs={"question": question})
        print("\nSupport:", result)