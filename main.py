from agents import Runner, Agent, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig 
from openai.types.responses import ResponseTextDeltaEvent

import os 
from dotenv import load_dotenv 
import chainlit as cl
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
   api_key=gemini_api_key,
   base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    
    )
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
   
)
Config=RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
)

agent = Agent(
    name="Teacher", 
    description="a teacher who can help you with your homework any quarryes you have and answer any question you have",
    instructions="your are a professional Teacher",
    

)
 
@cl.on_chat_start
async def handle_start():
     cl.user_session.set("history", [])
     await cl.Message(content="Hello, From mmbrother how can I help you?").send()


@cl.on_message
async def handle_message(message : cl.Message):
   history = cl.user_session.get("history")

   history.append({"role":"user","content":message.content})
   msg =cl.Message(content="")
   await msg.send()
   
   result =  Runner.run_streamed(
      agent,
      input=history,
      run_config=Config
      
   )
   async for event in result.stream_events():
       if event.type =="raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
           await msg.stream_token(event.data.delta)


   history.append({"role":"assistant","content":result.final_output}) 
   cl.user_session.set("history", history)  