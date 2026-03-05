"""
AI Content Generator - Gradio Web UI
Run locally on Mac M1/M2/M3 or with NVIDIA GPU
"""
import gradio as gr
import torch
import gc
from pathlib import Path

# Import generators
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from image_generator import ImageGenerator
from video_generator import VideoGenerator
from audio_generator import AudioGenerator, SimpleAudioGenerator

# Global instances
image_gen = None
video_gen = None
audio_gen = None


def get_device_info():
    """Get current device info"""
    if torch.cuda.is_available():
        return f"🖥️ CUDA GPU: {torch.cuda.get_device_name(0)}"
    elif torch.backends.mps.is_available():
        return "🍎 Apple Silicon (MPS)"
    return "💻 CPU (slower)"


def clear_memory():
    """Clear GPU/MPS memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()


def generate_image_only(prompt, steps, seed):
    """Generate just an image"""
    global image_gen

    if not prompt.strip():
        return None, "❌ Please enter a prompt"

    try:
        if image_gen is None:
            yield None, "Loading image model..."
            image_gen = ImageGenerator()

        yield None, "Generating image..."

        seed_val = int(seed) if seed else None
        image = image_gen.generate(prompt, num_steps=int(steps), seed=seed_val)

        # Save
        output_path = "outputs/generated_image.png"
        image_gen.save(image, output_path)

        yield image, "✅ Image generated!"

    except Exception as e:
        yield None, f"❌ Error: {str(e)}"


def generate_video_only(prompt, frames, steps):
    """Generate a video"""
    global video_gen

    if not prompt.strip():
        return None, "❌ Please enter a prompt"

    try:
        if video_gen is None:
            yield None, "Loading video model (this takes a while)..."
            video_gen = VideoGenerator()

        yield None, f"Generating {frames}-frame video..."

        output_path = "outputs/generated_video.mp4"
        video_gen.generate(
            prompt,
            num_frames=int(frames),
            num_steps=int(steps),
            output_path=output_path
        )

        clear_memory()
        yield output_path, "✅ Video generated!"

    except Exception as e:
        yield None, f"❌ Error: {str(e)}"


def generate_audio_only(text, use_bark):
    """Generate audio from text"""
    global audio_gen

    if not text.strip():
        return None, "❌ Please enter text"

    try:
        if use_bark:
            yield None, "Loading Bark TTS (requires GPU)..."
            audio_gen = AudioGenerator()
            output_path = "outputs/generated_audio.wav"
        else:
            yield None, "Using Edge TTS (fast, CPU)..."
            audio_gen = SimpleAudioGenerator()
            output_path = "outputs/generated_audio.mp3"

        yield None, "Generating speech..."
        audio_gen.generate(text, output_path)

        yield output_path, "✅ Audio generated!"

    except Exception as e:
        yield None, f"❌ Error: {str(e)}"


def full_pipeline(visual_prompt, narration, gen_video, bark_tts, progress=gr.Progress()):
    """Run the full pipeline"""
    results = {"image": None, "video": None, "audio": None, "status": ""}

    if not visual_prompt.strip():
        return None, None, None, "❌ Please enter a visual prompt"

    try:
        # Create output directory
        Path("outputs").mkdir(exist_ok=True)

        # Step 1: Image
        progress(0.1, "Generating image...")
        img_result = None
        for img, status in generate_image_only(visual_prompt, 4, None):
            img_result = img
        results["image"] = img_result

        # Step 2: Video (optional)
        if gen_video:
            progress(0.3, "Generating video...")
            clear_memory()
            vid_result = None
            for vid, status in generate_video_only(visual_prompt, 16, 25):
                vid_result = vid
            results["video"] = vid_result

        # Step 3: Audio (optional)
        if narration.strip():
            progress(0.7, "Generating audio...")
            clear_memory()
            aud_result = None
            for aud, status in generate_audio_only(narration, bark_tts):
                aud_result = aud
            results["audio"] = aud_result

        progress(1.0, "Done!")
        results["status"] = "✅ All content generated!"

        return results["image"], results["video"], results["audio"], results["status"]

    except Exception as e:
        return None, None, None, f"❌ Error: {str(e)}"


# Build the UI
def create_ui():
    with gr.Blocks(
        title="AI Content Generator",
        theme=gr.themes.Soft(),
        css=".gradio-container { max-width: 1200px !important; }"
    ) as demo:
        gr.Markdown("# 🎨 AI Content Generator MVP")
        gr.Markdown(f"**Device:** {get_device_info()}")

        with gr.Tabs():
            # Tab 1: Full Pipeline
            with gr.TabItem("🚀 Full Pipeline"):
                with gr.Row():
                    with gr.Column():
                        full_prompt = gr.Textbox(
                            label="Visual Prompt",
                            placeholder="A serene lake at sunset with mountains in the background...",
                            lines=3
                        )
                        full_narration = gr.Textbox(
                            label="Narration Text (optional)",
                            placeholder="Welcome to nature's paradise...",
                            lines=2
                        )
                        with gr.Row():
                            full_gen_video = gr.Checkbox(label="Generate Video", value=False)
                            full_bark = gr.Checkbox(label="Use Bark TTS (GPU)", value=False)
                        full_btn = gr.Button("🚀 Generate All", variant="primary")
                        full_status = gr.Textbox(label="Status", interactive=False)

                    with gr.Column():
                        full_image = gr.Image(label="Generated Image")
                        full_video = gr.Video(label="Generated Video")
                        full_audio = gr.Audio(label="Generated Audio")

                full_btn.click(
                    full_pipeline,
                    inputs=[full_prompt, full_narration, full_gen_video, full_bark],
                    outputs=[full_image, full_video, full_audio, full_status]
                )

            # Tab 2: Image Only
            with gr.TabItem("🖼️ Image Only"):
                with gr.Row():
                    with gr.Column():
                        img_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="A cyberpunk cityscape with neon lights...",
                            lines=3
                        )
                        img_steps = gr.Slider(1, 8, value=4, step=1, label="Steps (4 is fast)")
                        img_seed = gr.Number(label="Seed (optional)", precision=0)
                        img_btn = gr.Button("Generate Image", variant="primary")
                        img_status = gr.Textbox(label="Status", interactive=False)

                    with gr.Column():
                        img_output = gr.Image(label="Generated Image")

                img_btn.click(
                    generate_image_only,
                    inputs=[img_prompt, img_steps, img_seed],
                    outputs=[img_output, img_status]
                )

            # Tab 3: Video Only
            with gr.TabItem("🎬 Video Only"):
                with gr.Row():
                    with gr.Column():
                        vid_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="A cat playing with a ball in a sunny garden...",
                            lines=3
                        )
                        vid_frames = gr.Slider(8, 32, value=16, step=8, label="Frames")
                        vid_steps = gr.Slider(10, 50, value=25, step=5, label="Steps")
                        vid_btn = gr.Button("Generate Video", variant="primary")
                        vid_status = gr.Textbox(label="Status", interactive=False)

                    with gr.Column():
                        vid_output = gr.Video(label="Generated Video")

                gr.Markdown("⚠️ Video generation requires ~8GB+ VRAM and takes several minutes")

                vid_btn.click(
                    generate_video_only,
                    inputs=[vid_prompt, vid_frames, vid_steps],
                    outputs=[vid_output, vid_status]
                )

            # Tab 4: Audio Only
            with gr.TabItem("🔊 Audio Only"):
                with gr.Row():
                    with gr.Column():
                        aud_text = gr.Textbox(
                            label="Text to Speak",
                            placeholder="Hello, this is an AI generated voice...",
                            lines=4
                        )
                        aud_bark = gr.Checkbox(
                            label="Use Bark (high quality, requires GPU)",
                            value=False
                        )
                        aud_btn = gr.Button("Generate Audio", variant="primary")
                        aud_status = gr.Textbox(label="Status", interactive=False)

                    with gr.Column():
                        aud_output = gr.Audio(label="Generated Audio")

                gr.Markdown("💡 Uncheck Bark to use Edge TTS (faster, CPU-based)")

                aud_btn.click(
                    generate_audio_only,
                    inputs=[aud_text, aud_bark],
                    outputs=[aud_output, aud_status]
                )

        gr.Markdown("---")
        gr.Markdown("### 💡 Tips")
        gr.Markdown("""
        - **Mac M1/M2/M3**: Image generation works well, video may be slow
        - **Image**: Uses SDXL-Turbo (4 steps = ~2-5 seconds)
        - **Video**: Uses ModelScope, requires good GPU
        - **Audio**: Edge TTS is fast & free, Bark is higher quality but needs GPU
        """)

    return demo


if __name__ == "__main__":
    # Create outputs folder
    Path("outputs").mkdir(exist_ok=True)

    # Launch
    demo = create_ui()
    demo.launch(
        share=False,  # Set True to get public URL
        server_name="0.0.0.0",
        server_port=7860
    )
