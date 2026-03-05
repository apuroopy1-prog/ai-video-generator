import gradio as gr
import torch
import subprocess
import os

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model
print("Loading image model...")
from diffusers import AutoPipelineForText2Image

image_pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    variant="fp16" if device == "cuda" else None
)
image_pipe.to(device)
print("Ready!")


def generate_image(prompt):
    if not prompt:
        return None
    image = image_pipe(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
    return image


def generate_audio(text, voice):
    if not text:
        return None
    output_path = "/tmp/output.mp3"
    text_clean = text.replace('"', "'")
    cmd = f'edge-tts --voice "{voice}" --text "{text_clean}" --write-media {output_path}'
    subprocess.run(cmd, shell=True, capture_output=True)
    if os.path.exists(output_path):
        return output_path
    return None


# Simple UI
with gr.Blocks(title="AI Content Generator") as demo:
    gr.Markdown("# 🎨 AI Content Generator")
    gr.Markdown("Create images and audio from text!")

    with gr.Tab("🖼️ Image"):
        img_prompt = gr.Textbox(label="Prompt", placeholder="A sunset over mountains...")
        img_btn = gr.Button("Generate Image", variant="primary")
        img_output = gr.Image(label="Result")

        gr.Examples(
            examples=["A sunset over mountains", "A cute robot", "Cyberpunk city at night"],
            inputs=img_prompt
        )

        img_btn.click(generate_image, inputs=img_prompt, outputs=img_output)

    with gr.Tab("🔊 Audio"):
        aud_text = gr.Textbox(label="Text", placeholder="Hello world!", lines=3)
        aud_voice = gr.Dropdown(
            choices=["en-US-AriaNeural", "en-US-GuyNeural", "en-GB-SoniaNeural",
                     "fr-FR-DeniseNeural", "es-ES-ElviraNeural", "de-DE-KatjaNeural"],
            value="en-US-AriaNeural",
            label="Voice"
        )
        aud_btn = gr.Button("Generate Audio", variant="primary")
        aud_output = gr.Audio(label="Result")

        aud_btn.click(generate_audio, inputs=[aud_text, aud_voice], outputs=aud_output)

    gr.Markdown("---")
    gr.Markdown("Built with SDXL-Turbo & Edge TTS")

demo.launch()
