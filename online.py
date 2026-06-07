from google.colab import userdata
from openai import OpenAI
import gradio as gr

# OpenRouter API Key

api_key = userdata.get("ResearchForge")

client = OpenAI(
base_url="https://openrouter.ai/api/v1",
api_key=api_key,
)

SYSTEM_PROMPT = """
You are ResearchForge AI, an offline academic publication assistant specialized in converting raw research information into IEEE-compliant publication assets.

Your primary objective is NOT to invent research.
Your responsibility is to structure, organize, standardize, format, visualize, and document the research information provided by the user.

Core Rules:

* Never fabricate experimental results.
* Never invent references unless explicitly requested.
* Never alter the meaning of the user's research.
* Maintain academic and professional writing style.
* Follow IEEE paper structure and formatting conventions.
* Produce publication-ready content.
* Preserve confidentiality of all research information.
* Generate outputs suitable for direct use in Overleaf.

Always generate:

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

      messages = [
          {
              "role": "system",
              "content": SYSTEM_PROMPT
          }
      ]

      for user_msg, bot_msg in history:
          messages.append({
              "role": "user",
              "content": user_msg
          })

          messages.append({
              "role": "assistant",
              "content": bot_msg
          })

      messages.append({
          "role": "user",
          "content": message
      })

      completion = client.chat.completions.create(
          model="google/gemma-4-31b-it:free",
          messages=messages,
          temperature=0.2
      )

      return completion.choices[0].message.content

  except Exception as e:
      return f"Error: {str(e)}"

iface = gr.ChatInterface(
fn=chatbot_interface,
title="ResearchForge AI",
description="Offline Research-to-IEEE Publication Assistant"
)

iface.launch(share=True)
