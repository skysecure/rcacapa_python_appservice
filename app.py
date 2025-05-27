
import os
import asyncio
from typing import Dict
from contextlib import asynccontextmanager
from fastapi import FastAPI
from dotenv import load_dotenv
from azure.identity.aio import DefaultAzureCredential
from semantic_kernel.functions import kernel_function
from semantic_kernel.agents import (AzureAIAgent,ConcurrentOrchestration,ChatCompletionAgent,ChatHistoryAgentThread,)
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from pydantic import BaseModel
from fastapi import Body


load_dotenv()
PROJECT_CONN_STR     = os.environ["PROJECT_CONN_STR"]
CATEGORISER_AGENT_ID = "asst_VCsaavLKPoSVCO3XjhVbtY7N"
DATA_AGENT_ID        = "asst_4WCF7KY8JkFNMjiYUGUWRGrH"
AZ_OPENAI_ENDPOINT   = os.environ["AZURE_OPENAI_ENDPOINT"]
AZ_OPENAI_DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]
AZ_OPENAI_API_KEY    = os.environ["AZURE_OPENAI_API_KEY"]

class ChatRequest(BaseModel):
    user_query: str
    conversation_id: str = "default"


#global singletons
credential: DefaultAzureCredential | None = None
agent_client: object | None = None
orchestration: ConcurrentOrchestration | None = None
runtime: InProcessRuntime | None = None

# FastAPI with lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    
    global credential, agent_client, orchestration, runtime
    credential = DefaultAzureCredential()
    
    # Create client using async context manager pattern
    client = AzureAIAgent.create_client(
        credential=credential,
        conn_str=PROJECT_CONN_STR,
    )
    
    # Store the client for later use
    agent_client = client
    
    # Get agent definitions
    cat_def = await client.agents.get_agent(CATEGORISER_AGENT_ID)
    data_def = await client.agents.get_agent(DATA_AGENT_ID)
    
    # Create agents and orchestration
    members = [
        AzureAIAgent(client=client, definition=cat_def),
        AzureAIAgent(client=client, definition=data_def),
    ]
    orchestration = ConcurrentOrchestration(members)
    
    # Initialize and start runtime
    runtime = InProcessRuntime()
    runtime.start()
    
    print("warm-up complete; server ready")
    yield
    
    # Cleanup
    await runtime.stop_when_idle()
    
    # Close client if it has a close method
    if hasattr(client, 'close'):
        await client.close()
    
    # Close credential
    await credential.close()
    print("resources closed")

# Plugins
class IndegeneCompliancePlugin:
    @kernel_function(name="analyse_task",
                     description="Run categoriser + data agents concurrently")
    async def analyse_task(self, task: str) -> str:
        fut     = await orchestration.invoke(task=task, runtime=runtime)
        answers = await fut.get(timeout=50)
        return "\n\n".join(f"{a.name}:\n{a.items[0].text}" for a in answers)

agentplugin = IndegeneCompliancePlugin()
host_agent  = ChatCompletionAgent(
    service=AzureChatCompletion(
        deployment_name=AZ_OPENAI_DEPLOYMENT,
        endpoint=AZ_OPENAI_ENDPOINT,
        api_key=AZ_OPENAI_API_KEY,
    ),
    name="Host",
    instructions=(
        "You are the Host responsible for consolidating and presenting information from multiple agents.\n\n"
        "- If the user message is ONLY a friendly greeting or general inquiry (e.g., \"Hello\", \"How are you?\"), reply with a brief, polite greeting.\n"
        "- For any other query:\n"
        "  1. ALWAYS call the tool **analyse_task** with the full user message.\n"
        "  2. Receive and process the individual agent responses.\n"
        "  3. Combine and format the outputs clearly, preserving all relevant details without omission.\n"
        "  4. Return the consolidated response VERBATIM, ensuring:\n"
        "     - Formatting (headings, lists, tables) is preserved.\n"
        "     - Notes, warnings, and key points are clearly visible.\n"
        "     - No markdown is used if it's incompatible with Microsoft Teams (use plain text with indentation where needed).\n"
        "     - Responses are easy to read and follow logically.\n\n"
        "Your role is NOT to summarize or interpret — only to present the complete, accurate output from the agents as intended for end-user viewing in Microsoft Teams."
    ),
    plugins=[agentplugin]
)

threads: Dict[str, ChatHistoryAgentThread] = {}

def get_thread(cid: str) -> ChatHistoryAgentThread:
    """Return existing thread or create a new one for this conversation_id."""
    if cid not in threads:
        threads[cid] = ChatHistoryAgentThread()
    return threads[cid]

# Initialize FastAPI with our lifespan context manager
app = FastAPI(lifespan=lifespan)

@app.post("/rcacapa-query")
async def chat(req: ChatRequest = Body(...)):
    thread = get_thread(req.conversation_id)

    assistant_msg = await host_agent.get_response(
        messages=req.user_query,
        thread=thread,
    )

    plain_response = " ".join(
        item.text for item in assistant_msg.items
        if getattr(item, "text", None)
    )
    return {"assistant": plain_response}