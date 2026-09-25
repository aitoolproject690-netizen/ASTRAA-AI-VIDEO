import os
import tempfile
import gradio as gr
import torch
from diffusers import LTXPipeline
from diffusers.utils import export_to_video
from PIL import Image

MODEL_ID = "Lightricks/LTX-Video"
pipe = None

def load_pipe():
    global pipe
    if pipe is None:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        pipe = LTXPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
        if torch.cuda.is_available():
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cpu")
    return pipe

def generate(image, prompt, frames, width, height):
    if image is None:
        raise gr.Error("A reference image is required.")
    if not prompt.strip():
        raise gr.Error("Prompt is required.")
    p = load_pipe()
    image = image.convert("RGB")
    result = p(
        prompt=prompt,
        image=image,
        width=int(width),
        height=int(height),
        num_frames=int(frames),
        guidance_scale=3.0,
        num_inference_steps=8,
    )
    frames_out = result.frames[0]
    path = os.path.join(tempfile.gettempdir(), "astraa_shot.mp4")
    export_to_video(frames_out, path, fps=24)
    return path

with gr.Blocks(title="ASTRAA AI Video") as demo:
    gr.Markdown("# 🎬 ASTRAA — AI Video")
    gr.Markdown("LTX-Video image-to-video. Generate short cinematic shots from ASTRAA character references.")
    with gr.Row():
        image = gr.Image(type="pil", label="Reference image")
        with gr.Column():
            prompt = gr.Textbox(
                label="Shot prompt",
                value="Aarav and his mother remain visually consistent with the reference image. Cinematic 3D Indian fantasy movie style, subtle natural movement, dramatic lighting, gentle camera push-in, realistic motion, no text."
            )
            frames = gr.Slider(9, 33, value=17, step=8, label="Frames")
            width = gr.Dropdown([256, 384, 512], value=384, label="Width")
            height = gr.Dropdown([256, 384, 512], value=256, label="Height")
            generate_btn = gr.Button("Generate video", variant="primary")
    output = gr.Video(label="Generated shot")
    generate_btn.click(generate, [image, prompt, frames, width, height], output)

demo.queue().launch()
