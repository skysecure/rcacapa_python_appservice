import os
import asyncio
from typing import Dict, List
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from azure.identity.aio import DefaultAzureCredential
from semantic_kernel.functions import kernel_function
from semantic_kernel.agents import (AzureAIAgent,ConcurrentOrchestration,ChatCompletionAgent,ChatHistoryAgentThread,)
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from pydantic import BaseModel
from fastapi import Body

load_dotenv()

class ChatRequest(BaseModel):
    user_query: str
    conversation_id: str = "default"


PROJECT_CONN_STR     = os.environ["PROJECT_CONN_STR"]
AZ_OPENAI_ENDPOINT   = os.environ["AZURE_OPENAI_ENDPOINT"]
AZ_OPENAI_DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]
AZ_OPENAI_API_KEY    = os.environ["AZURE_OPENAI_API_KEY"]

#Agents
CATEGORISER_AGENT_ID = "asst_CzSz70pLt0mpDWkYDtiD2021"
DATA_AGENT_ID        = "asst_z6Pw3Zn3nIC6SgrELiqwLoLj"

# Helper to build the two member agents
async def _build_members(client) -> List[AzureAIAgent]:
    cat_def  = await client.agents.get_agent(CATEGORISER_AGENT_ID)
    data_def = await client.agents.get_agent(DATA_AGENT_ID)
    return [
        AzureAIAgent(client=client, definition=cat_def),
        AzureAIAgent(client=client, definition=data_def),
    ]

class IndegeneCompliancePlugin:
    @kernel_function(
        name="analyse_task",
        description="Run categoriser + data agents concurrently; return their answers.",
    )
    async def analyse_task(self, task: str) -> str:
        credential = DefaultAzureCredential()
        async with credential:
            async with AzureAIAgent.create_client(
                credential=credential, conn_str=PROJECT_CONN_STR
            ) as client:
                members       = await _build_members(client)
                orchestration = ConcurrentOrchestration(members)
                runtime       = InProcessRuntime()
                runtime.start()

                try:
                    fut     = await orchestration.invoke(task=task, runtime=runtime)
                    answers = await fut.get(timeout=50)
                    # answers[i].items[0].text → plain response text
                    return "\n\n".join(f"{a.name}:\n{a.items[0].text}" for a in answers)
                finally:
                    await runtime.stop_when_idle()


agentplugin      = IndegeneCompliancePlugin()
host_agent  = ChatCompletionAgent(
    service=AzureChatCompletion(
        deployment_name=AZ_OPENAI_DEPLOYMENT,
        endpoint=AZ_OPENAI_ENDPOINT,
        api_key=AZ_OPENAI_API_KEY,
    ),
    name="Host",
    instructions=(
        "You are a helpful assistant.\n\n"
        "• If the user message is ONLY a friendly greeting or general inquiry, such as reply with a brief greeting.\n"
        "• Otherwise, ALWAYS call the tool **analyse_task** with the "
            "full user message, then return the full tool result verbatim."
    ),
    plugins=[agentplugin],
)

threads: Dict[str, ChatHistoryAgentThread] = {}

def get_thread(cid: str) -> ChatHistoryAgentThread:
    """Return existing thread or create a new one for this conversation_id."""
    if cid not in threads:
        threads[cid] = ChatHistoryAgentThread()
    return threads[cid]

app = FastAPI()

@app.post("/rcacapa-query")
async def chat(req: ChatRequest = Body(...)):
    thread = get_thread(req.conversation_id)

    assistant_msg = await host_agent.get_response(
        messages=req.user_query,
        thread=thread,
    )

    plain_response = " ".join(
        item.text for item in assistant_msg.items
        if getattr(item, "text", None)          # ignore tool-call items
    )
    return {"assistant": plain_response}


@app.post("/rcacapa-query/stream_plain")
async def chat_stream_plain(req: ChatRequest = Body(...)):
    thread = get_thread(req.conversation_id)

    async def gen():
        async for chunk in host_agent.invoke_stream(
            messages=req.user_query,
            thread=thread,
        ):
            txt = getattr(chunk, "content", None)
            if txt:
                yield txt

    return StreamingResponse(gen(), media_type="text/plain")