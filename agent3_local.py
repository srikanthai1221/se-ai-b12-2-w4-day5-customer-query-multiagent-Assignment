from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from local_tool import lookup_order

load_dotenv()
llm = LLM(model="gpt-4o-mini", temperature=0.2)

local_agent = Agent(
    role="Order Records Specialist",
    goal="Give customers accurate information about their own orders from ShopEase's local records",
    backstory=(
        "You work in ShopEase customer support with access to the order records. "
        "You always look up the order before answering and never guess an order's status."
    ),
    tools=[lookup_order],
    llm=llm,
    verbose=True,
)

local_task = Task(
    description=(
        "Customer question: {question}\n"
        "If the question has an order ID, look it up and answer using the record. "
        "If there is no order ID in the question, reply exactly: NEED_ORDER_ID. "
        "If the order is not found, say so politely."
    ),
    expected_output="A short answer about the customer's order, NEED_ORDER_ID, or a polite 'order not found' message.",
    agent=local_agent,
)

if __name__ == "__main__":
    crew = Crew(agents=[local_agent], tasks=[local_task], verbose=True)
    result = crew.kickoff(inputs={"question": "Where is my order?"})
    print("\nFINAL ANSWER:\n", result)