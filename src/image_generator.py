"""
Text-to-Image Generator Module
Works on Mac M1/M2/M3 with MPS acceleration
"""
import torch
from diffusers import AutoPipelineForText2Image
from pathlib import Path


class ImageGenerator:
    def __init__(self, model_id: str = "stabilityai/sdxl-turbo"):
        self.model_id = model_id
        self.pipe = None
        self.device = self._get_device()

    def _get_device(self) -> str:
        """Get the best available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
        return "cpu"

    def load(self):
        """Load the model"""
        if self.pipe is not None:
            return

        print(f"Loading {self.model_id} on {self.device}...")

        # Use float16 for GPU, float32 for CPU
        dtype = torch.float16 if self.device != "cpu" else torch.float32

        self.pipe = AutoPipelineForText2Image.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None
        )
        self.pipe.to(self.device)
        print("✅ Image model loaded!")

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_steps: int = 4,
        guidance_scale: float = 0.0,
        width: int = 512,
        height: int = 512,
        seed: int = None
    ):
        """Generate an image from a text prompt"""
        self.load()

        # Set seed for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(self.device).manual_seed(seed)

        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator
        ).images[0]

        return image

    def save(self, image, path: str):
        """Save image to file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        print(f"✅ Saved to {path}")
        return path


# Standalone usage
if __name__ == "__main__":
    generator = ImageGenerator()

    prompt = "A beautiful sunset over mountains, cinematic lighting, 8k, detailed"
    image = generator.generate(prompt)
    generator.save(image, "outputs/generated_image.png")
