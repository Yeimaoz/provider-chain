"""Chinese summarization with multi-provider fallback."""
from lcz_llm_chain import ask

article = "..."  # 任意中文文章
summary = ask("summary_zh", f"請用 100 字總結以下內容：\n\n{article}", max_tokens=200)
print(summary)
