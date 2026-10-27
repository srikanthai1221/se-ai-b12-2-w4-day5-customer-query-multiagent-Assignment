import csv
import os
from crewai.tools import tool

ORDERS_PATH = os.path.join(os.path.dirname(__file__), "data", "orders.csv")


@tool("Local Order Lookup")
def lookup_order(order_id: str) -> str:
    """Look up a customer's order in ShopEase's local order records using the order ID
    (for example ORD1001). Returns the product, order status and delivery details."""
    order_id = order_id.strip().upper()
    with open(ORDERS_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["order_id"] == order_id:
                # Leave out the email so personal data isn't passed to the LLM
                return ", ".join(f"{k}: {v}" for k, v in row.items() if k != "email")
    return f"No order found with ID {order_id}."