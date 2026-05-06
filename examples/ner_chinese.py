"""Chinese NER example using gemini → groq → openrouter chain."""
from provider_chain import ask

prompt = """Extract structured info from the news, return JSON:
{
  "tickers": [],
  "sentiment_score": 0.0,
  "importance": 0.0
}

Title: 鴻海 Q1 EPS 創高
Body: 鴻海 (2317) 公布 2026 第一季財報，營收 1.5 兆元年增 8%，淨利 487 億元。
"""

result = ask("ner", prompt, max_tokens=512, temperature=0.0)
print(result)
