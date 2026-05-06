"""Demonstrate which provider answered."""
from provider_chain import ask
from provider_chain.chain import LAST_USED

result = ask("ner", "鴻海 (2317) Q1 EPS 創高", max_tokens=100)
print(f"Result: {result}")
print(f"Answered by: {LAST_USED['provider']} / {LAST_USED['model']}")
