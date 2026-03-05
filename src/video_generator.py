"""
Text-to-Video Generator Module
Uses ModelScope for video generation
"""
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
from pathlib import Path


class VideoGenerator:
    def __init__(self, model_id: str = "damo-vilab/text-to-video-ms-1.7b"):
        self.model_id = model_id
        self.pipe = None
        self.device = self._get_device()

    def _get_device(self) -> str:
        """Get the best available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            # Note: Video models may have limited MPS support
            return "mps"
        return "cpu"

    def load(self):
        """Load the model"""
        if self.pipe is not None:
            return

        print(f"Loading {self.model_id}...")
        print("⚠️  This requires ~8GB+ VRAM/RAM")

        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.pipe = DiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None
        )

        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_vae_slicing()
        else:
            self.pipe.to(self.device)

        print("✅ Video model loaded!")

    def generate(
        self,
        prompt: str,
        num_frames: int = 16,
        num_steps: int = 25,
        fps: int = 8,
        width: int = 256,
        height: int = 256,
        output_path: str = "outputs/generated_video.mp4"
    ) -> str:
        """Generate a video from a text prompt"""
        self.load()

        print(f"Generating {num_frames} frames... (this may take a few minutes)")

        video_frames = self.pipe(
            prompt=prompt,
            num_frames=num_frames,
            num_inference_steps=num_steps,
            width=width,
            height=height
        ).frames[0]

        # Export to file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        export_to_video(video_frames, output_path, fps=fps)

        print(f"✅ Video saved to {output_path}")
        return output_path

    def clear_memory(self):
        """Clear GPU memory"""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Standalone usage
if __name__ == "__main__":
    generator = VideoGenerator()

    prompt = "A cat walking in a garden, high quality"
    video_path = generator.generate(prompt)
    print(f"Video created: {video_path}")
