import argparse
from src.generator import BaseLLM

from src.generator import OpenAILLM
from src.generator import GroqLLM
from src.generator import GeminiLLM

# build a parser with all models' args
parser = argparse.ArgumentParser()
OpenAILLM.add_args(parser)
GroqLLM.add_args(parser)
GeminiLLM.add_args(parser)
args = parser.parse_args()  # empty list → all defaults, no API calls made
# args = parser.parse_args([])  # empty list → all defaults, no API calls made

# check registry
print("Registered models:", BaseLLM.registry)

# instantiate each model (no generate/chat called)
# openai_llm = OpenAILLM(args)
groq_llm = GroqLLM(args)
# gemini_llm = GeminiLLM(args)

# print(f"OpenAI → model: {openai_llm.model_name}, params: {openai_llm.params}")
print(f"Groq   → model: {groq_llm.model_name},   params: {groq_llm.params}")
# print(f"Gemini → model: {gemini_llm.model_name}, params: {gemini_llm.params}")

print(groq_llm.generate("hello, how are you?"))


