def generate_followups(llm, last_answer):
    prompt = f"""
You are an assistant that suggests short follow-up questions.

Based on the answer below, generate 3 concise follow-up questions.

Rules:
- Each question under 12 words
- No numbering
- No markdown
- One question per line

Answer:
{last_answer}
"""

    response = llm.invoke(prompt)

    # ✅ Handle all LangChain return types safely
    if hasattr(response, "content"):
        text = response.content
    else:
        text = str(response)

    return [
        q.strip("-• ").strip()
        for q in text.split("\n")
        if q.strip()
    ]
