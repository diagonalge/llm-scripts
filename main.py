from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from typing import Optional
from loader import Loader

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_loader = Loader()

@app.post("/generate")
def infer(
    model_id: str = Form(),
    character_name: str = Form(),
    persona: str = Form(),
    prompt: str = Form(),
    chat_history: Optional[str] = Form(None)
):
    # pipe = _loader.get_pipe(Loader.Module.TEXT_TO_IMAGE, model)
    # img = inference(prompt, model, neg_prompt, cfg, seed, steps, aspect_ratio, pipe)
    # return Response(img)
    pass
