import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
try:
    from rag_api.sme_agent import SMEAgent
except ImportError as e:
    print(f"Error: Could not import the SMEAgent class. Please ensure that:")
    print(f"1. This script is in your project's root directory.")
    print(f"2. The agent's code is in a sub-directory (e.g., 'rag_api/sme_agent.py').")
    print(f"3. An empty '__init__.py' file exists in the 'rag_api' folder.")
    print(f"Details: {e}")
    sys.exit(1)

if __name__ == "__main__":
    print("--- Initializing the SME Agent ---")
    sme_agent = SMEAgent()
    print("-" * 30)
    print("\n--- Running Example 1: Question Answering ---")
    query_1 = "What is the difference between hinge loss and logistic loss?"
    response_1 = sme_agent.run(query_1)
    
    print("\n[Final Answer]")
    print(response_1.get("output", "No output generated."))
    print("-" * 30)

    print("\n--- Running Example 2: Multi-step Content Creation and Saving ---")
    query_2 = "Create an assignment on binary classification and save it as 'assignment.pdf'."
    response_2 = sme_agent.run(query_2)
    
    print("\n[Final Answer]")
    print(response_2.get("output", "No output generated."))
    print("-" * 30)


