from dotenv import load_dotenv
import os
import anthropic

# Load your API key from .env
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")

# Create the client
client = anthropic.Anthropic(api_key=api_key)

# Send a message
message = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello Claude! Say hi back in one sentence."}
    ]
)

# Print the response
print(message.content[0].text)