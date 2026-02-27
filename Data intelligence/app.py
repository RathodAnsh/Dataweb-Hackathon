import chainlit as cl
from agent_logic import create_agent
import os
import glob

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("agent_created", False)
    await cl.Message(
        content="👋 Welcome! Please upload a CSV or Excel file to begin analysis."
    ).send()


@cl.on_message
async def on_message(message: cl.Message):

    # Detect file upload
    file = next((f for f in message.elements if "application" in f.mime or "text" in f.mime), None)
    agent_created = cl.user_session.get("agent_created")

    # If file uploaded and agent not yet created
    if file and not agent_created:
        await cl.Message(content=f"📂 Processing `{file.name}`...").send()

        try:
            # UPDATED: Capture both the agent and the data preview markdown
            agent, preview_html = create_agent(file.path)

            cl.user_session.set("agent", agent)
            cl.user_session.set("agent_created", True)

            # NEW: Show data preview immediately after upload
            await cl.Message(
                content=f"✅ **Dataset loaded successfully!**\n\n### Data Preview:\n{preview_html}\n\nYou can now ask questions about your data."
            ).send()

        except Exception as e:
            await cl.Message(content=f"❌ Error processing file: {str(e)}").send()

        return

    # If user asks question before uploading file
    if not agent_created:
        await cl.Message(content="⚠️ Please upload a CSV or Excel file first.").send()
        return

    # Clear old charts before processing new query
    if os.path.exists("chart.png"):
        try:
            os.remove("chart.png")
        except Exception:
            pass

    # Get agent
    agent = cl.user_session.get("agent")

    # Run AutoGen pipeline
    response = await cl.make_async(agent)(message.content)

    final_answer = response.get("output", "❌ I could not generate an answer.")
    has_viz = response.get("has_viz", False)

    # Attach Visualization if generated
    elements = []
    if has_viz and os.path.exists("chart.png"):
        image = cl.Image(path="chart.png", name="Data Visualization", display="inline")
        elements.append(image)

    # Send final message - Chainlit will render the ```sql block with a Copy button automatically
    await cl.Message(content=final_answer, elements=elements).send()