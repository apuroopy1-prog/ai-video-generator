#!/usr/bin/env python3
"""
Quick Start Script - Generate your first content!
Run: python quick_start.py
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    print("🎨 AI Content Generator - Quick Start")
    print("=" * 50)

    # Create outputs folder
    Path("outputs").mkdir(exist_ok=True)

    # Check device
    import torch
    if torch.cuda.is_available():
        device = f"CUDA ({torch.cuda.get_device_name(0)})"
    elif torch.backends.mps.is_available():
        device = "Apple Silicon (MPS)"
    else:
        device = "CPU"
    print(f"Device: {device}\n")

    # Menu
    print("What would you like to generate?")
    print("1. Image only (fastest)")
    print("2. Image + Audio")
    print("3. Full pipeline (Image + Video + Audio)")
    print("4. Launch Web UI")
    print("0. Exit")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == "1":
        generate_image()
    elif choice == "2":
        generate_image_audio()
    elif choice == "3":
        generate_full()
    elif choice == "4":
        launch_ui()
    else:
        print("Goodbye!")
        return


def generate_image():
    """Generate a single image"""
    from image_generator import ImageGenerator

    prompt = input("Enter prompt (or press Enter for default): ").strip()
    if not prompt:
        prompt = "A beautiful sunset over mountains, cinematic lighting, 8k"

    print(f"\nGenerating image for: '{prompt}'")
    gen = ImageGenerator()
    image = gen.generate(prompt)
    path = gen.save(image, "outputs/quick_image.png")
    print(f"\n✅ Image saved to: {path}")


def generate_image_audio():
    """Generate image with audio narration"""
    from image_generator import ImageGenerator
    from audio_generator import SimpleAudioGenerator

    prompt = input("Enter image prompt (or Enter for default): ").strip()
    if not prompt:
        prompt = "A serene lake at sunset with mountains"

    narration = input("Enter narration text (or Enter for default): ").strip()
    if not narration:
        narration = "Welcome to nature's paradise. A peaceful lake reflects the golden sunset."

    print(f"\n1/2 Generating image...")
    img_gen = ImageGenerator()
    image = img_gen.generate(prompt)
    img_gen.save(image, "outputs/quick_image.png")

    print(f"2/2 Generating audio...")
    aud_gen = SimpleAudioGenerator()
    aud_gen.generate(narration, "outputs/quick_audio.mp3")

    print("\n✅ Done!")
    print("   Image: outputs/quick_image.png")
    print("   Audio: outputs/quick_audio.mp3")


def generate_full():
    """Full pipeline"""
    from pipeline import ContentPipeline

    prompt = input("Enter visual prompt (or Enter for default): ").strip()
    if not prompt:
        prompt = "A cat walking in a sunny garden"

    narration = input("Enter narration (or Enter for default): ").strip()
    if not narration:
        narration = "Watch this cute cat explore the beautiful garden."

    print("\n⚠️  Full pipeline takes several minutes. Continue? (y/n): ", end="")
    if input().strip().lower() != 'y':
        print("Cancelled.")
        return

    pipeline = ContentPipeline()
    results = pipeline.run(
        visual_prompt=prompt,
        narration_text=narration,
        generate_video=True,
        output_dir="outputs"
    )

    print("\n✅ All files generated in outputs/")


def launch_ui():
    """Launch Gradio UI"""
    print("\nLaunching web UI...")
    print("Open http://localhost:7860 in your browser\n")

    import subprocess
    subprocess.run([sys.executable, "app.py"])


if __name__ == "__main__":
    main()
