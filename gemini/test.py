from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv("gemini/.env")
# Set the Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

pick_fn_declaration = {
    "name": "pick",
    "description": "The prompt contains a list of objects on the table. Pick one of them according to the instruction from the user. " \
        "Use your knowledge to identify whether the object is a cylinder, cuboid, sphere or thin plate.",
    "parameters": {
        "type": "object",
        "properties": {
            "object": {
                "type": "string",
                "description": "The object to pick from the table. The object has to be chosen from the list of objects on the table."
            },
            "shape": {
                "type": "string",
                "description": "The shape of the object. The shape can be a cylinder, cuboid, sphere or thin plate."
            }
        },
        "required": ["object", "shape"],
    },
}

# Define client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
tools = types.Tool(function_declarations=[pick_fn_declaration])
config = types.GenerateContentConfig(tools=[tools])


prompt = "\nTable Items: refrigerator, box, vase, igniter, ball, tape measure, mug, cup, chair, poster, flashlight, earphone. \nPick up the mug. "
# Generate content using the model
response = client.models.generate_content(
    model="gemini-2.5-flash-preview-04-17",
    contents=prompt,
    config=config,
)
# Print the response
print("\n\nPrompt:\n"+prompt)
# Print the function call
print("\nResponse:\n" + str(response.function_calls) + "\n")
print(response.text)