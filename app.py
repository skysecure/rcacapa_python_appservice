from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import uvicorn
import traceback
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

# Request schema
class ChatRequest(BaseModel):
    user_query: str

# Initialize FastAPI
app = FastAPI()

# Initialize Azure AI Project Client
project_client = AIProjectClient.from_connection_string(
    credential=DefaultAzureCredential(),
    conn_str="eastus2.api.azureml.ms;a182fe30-ddfd-4f86-9bb2-9278c2f0c684;agents;zohoagent"
)

# Load agent
agent = project_client.agents.get_agent("asst_uMs6c90Z7hOdymAqcEqVvWZ5")


# Core chat handler with improved error resilience
def query_with_dynamic_thread(user_input: str) -> str:
    MAX_RETRIES = 3
    TIMEOUT_SECONDS = 20  # Slightly under Teams 15s safety window
    GRACE_RETRY_COUNT = 2
    RETRY_DELAY_SECONDS = 2

    for attempt in range(MAX_RETRIES):
        try:
            print(f"[Attempt {attempt + 1}] Creating thread and posting message...")

            thread = project_client.agents.create_thread()
            project_client.agents.create_message(
                thread_id=thread.id,
                role="user",
                content=user_input
            )

            run = project_client.agents.create_and_process_run(
                thread_id=thread.id,
                agent_id=agent.id
            )

            start_time = time.time()

            while True:
                elapsed = time.time() - start_time
                if elapsed > TIMEOUT_SECONDS:
                    raise TimeoutError("Timed out waiting for agent run to complete.")

                run_status = project_client.agents.get_run(
                    thread_id=thread.id,
                    run_id=run.id
                )

                if run_status.status == "completed":
                    break
                elif run_status.status in ["failed", "cancelled"]:
                    raise RuntimeError(f"Run failed with status: {run_status.status}")

                time.sleep(1)

            # Try to collect messages normally
            response = collect_assistant_response(thread.id)
            if response:
                return response
            else:
                raise RuntimeError("Run completed but no assistant response found.")

        except (TimeoutError, RuntimeError) as e:
            print(f"[Warning] Attempt {attempt + 1} failed: {e}")
            # Grace retry to pull message anyway
            for i in range(GRACE_RETRY_COUNT):
                print(f"[Grace Retry {i + 1}] Trying to pull response after failure...")
                fallback = collect_assistant_response(thread.id)
                if fallback:
                    return fallback
                time.sleep(1)

            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                return "Still processing your request. Please wait a few seconds and try again."

        except Exception as e:
            print(f"[Fatal] Unexpected error: {str(e)}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail="Unexpected internal server error.")

    return "Agent failed to respond after multiple attempts. Please try again shortly."


# Message collection helper
def collect_assistant_response(thread_id: str) -> str:
    try:
        messages = project_client.agents.list_messages(thread_id=thread_id)
        response_texts = []
        for msg in messages.data:
            if msg.role == "assistant":
                for content in msg.content:
                    if hasattr(content, "text") and content.text:
                        response_texts.append(content.text.value)
        return "\n".join(response_texts) if response_texts else None
    except Exception as e:
        print(f"[Error] Collecting message failed: {e}")
        return None


# API route
@app.post("/rcacapa-query")
async def chat(request: ChatRequest):
    try:
        start_time = time.time()
        response = query_with_dynamic_thread(request.user_query)
        duration = time.time() - start_time
        print(f"[INFO] Total processing time: {duration:.2f}s")
        return {"response": response}
    except Exception as e:
        print(f"[API ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run app (local dev)
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)