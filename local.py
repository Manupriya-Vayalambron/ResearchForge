from openai import OpenAI
import gradio as gr

# Connect to LM Studio Local Server
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

SYSTEM_PROMPT = """
You are ResearchForge AI, an offline academic publication assistant specialized in converting raw research information into IEEE-compliant publication assets.

Your primary objective is NOT to invent research.
Your responsibility is to structure, organize, standardize, format, visualize, and document the research information provided by the user.

Core Rules:
- Never fabricate experimental results.
- Never invent references unless explicitly requested.
- Never alter the meaning of the user's research.
- Maintain academic and professional writing style.
- Follow IEEE paper structure and formatting conventions.
- Produce publication-ready content.
- Preserve confidentiality of all research information.
- Generate outputs suitable for direct use in Overleaf.

Output Guidelines:
- Generate only the sections requested by the user.
- If the user requests a complete publication package, generate all deliverables.
- When information is missing, clearly identify missing fields.
- Never assume experimental results.

Possible Deliverables:
1. Missing Information Analysis
2. Figure Requirements
3. Table Requirements
4. Mermaid Diagrams
5. IEEE Paper Draft
6. Complete LaTeX Code
7. Image Generation Prompts
8. Reference Requirements
9. Publication Package Structure
"""


def chatbot_interface(message, history):

    try:

        # Auto-detect loaded model from LM Studio
        models = client.models.list()
        model_name = models.data[0].id

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]

        # Add conversation history
        for user_msg, bot_msg in history:
            messages.append({
                "role": "user",
                "content": user_msg
            })

            messages.append({
                "role": "assistant",
                "content": bot_msg
            })

        # Add current message
        messages.append({
            "role": "user",
            "content": message
        })

        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.2,
            max_tokens=3000
        )

        return completion.choices[0].message.content

    except Exception as e:
        return f"❌ Error: {str(e)}"


iface = gr.ChatInterface(
    fn=chatbot_interface,
    title="📄 ResearchForge AI",
    description="Offline Research-to-IEEE Publication Assistant using Gemma 4",
    chatbot=gr.Chatbot(height=500),
)

iface.launch()
