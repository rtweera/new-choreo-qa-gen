from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
import yaml

load_dotenv(verbose=True, override=True)

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

model = init_chat_model(
    model=config["model"]["id"],
    model_provider=config["model"]["provider"],
)

messages = [
    SystemMessage(content="Convert the following to Sinhala"),
    HumanMessage(content="How are you?"),
]

print(model.invoke(messages))