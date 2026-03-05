"""
Combined Pipeline: Text → Image → Video → Audio
"""
import gc
import torch
from pathlib import Path
from moviepy.editor import VideoFileClip, AudioFileClip


class ContentPipeline:
    """Full content generation pipeline"""

    def __init__(self):
        self.image_gen = None
        self.video_gen = None
        self.audio_gen = None

    def _clear_memory(self):
        """Clear GPU/MPS memory between generations"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def generate_image(self, prompt: str, output_dir: str = "outputs") -> str:
        """Generate image from prompt"""
        if self.image_gen is None:
            from image_generator import ImageGenerator
            self.image_gen = ImageGenerator()

        image = self.image_gen.generate(prompt)
        output_path = f"{output_dir}/image.png"
        self.image_gen.save(image, output_path)
        return output_path

    def generate_video(self, prompt: str, output_dir: str = "outputs") -> str:
        """Generate video from prompt"""
        if self.video_gen is None:
            from video_generator import VideoGenerator
            self.video_gen = VideoGenerator()

        output_path = f"{output_dir}/video.mp4"
        self.video_gen.generate(prompt, output_path=output_path)
        return output_path

    def generate_audio(self, text: str, output_dir: str = "outputs") -> str:
        """Generate audio from text"""
        if self.audio_gen is None:
            from audio_generator import AudioGenerator
            self.audio_gen = AudioGenerator()

        output_path = f"{output_dir}/audio.wav"
        self.audio_gen.generate(text, output_path=output_path)
        return output_path

    def combine_video_audio(
        self,
        video_path: str,
        audio_path: str,
        output_path: str = "outputs/final.mp4"
    ) -> str:
        """Combine video with audio track"""
        print("Combining video and audio...")

        video = VideoFileClip(video_path)
        audio = AudioFileClip(audio_path)

        # Adjust audio duration to match video
        if audio.duration > video.duration:
            audio = audio.subclip(0, video.duration)

        # Combine
        final_video = video.set_audio(audio)
        final_video.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            logger=None  # Suppress verbose output
        )

        # Cleanup
        video.close()
        audio.close()
        final_video.close()

        print(f"✅ Final video saved to {output_path}")
        return output_path

    def run(
        self,
        visual_prompt: str,
        narration_text: str = None,
        generate_video: bool = True,
        output_dir: str = "outputs"
    ) -> dict:
        """
        Run the full pipeline

        Args:
            visual_prompt: Prompt for image/video generation
            narration_text: Text for voice narration (optional)
            generate_video: Whether to generate video (slower)
            output_dir: Directory for outputs

        Returns:
            Dictionary with paths to generated files
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        results = {}

        # Step 1: Generate Image
        print("\n📸 Step 1: Generating image...")
        results['image'] = self.generate_image(visual_prompt, output_dir)
        self._clear_memory()

        # Step 2: Generate Video (optional)
        if generate_video:
            print("\n🎬 Step 2: Generating video...")
            results['video'] = self.generate_video(visual_prompt, output_dir)
            self._clear_memory()

        # Step 3: Generate Audio (if narration provided)
        if narration_text:
            print("\n🔊 Step 3: Generating audio...")
            results['audio'] = self.generate_audio(narration_text, output_dir)
            self._clear_memory()

            # Step 4: Combine video + audio
            if generate_video:
                print("\n🎞️ Step 4: Combining video + audio...")
                results['final'] = self.combine_video_audio(
                    results['video'],
                    results['audio'],
                    f"{output_dir}/final_with_audio.mp4"
                )

        print("\n✅ Pipeline complete!")
        print(f"📁 Outputs saved to: {output_dir}/")

        return results


# Standalone usage
if __name__ == "__main__":
    pipeline = ContentPipeline()

    results = pipeline.run(
        visual_prompt="A futuristic city at night with neon lights and flying cars",
        narration_text="Welcome to the city of tomorrow, where dreams become reality.",
        generate_video=True,
        output_dir="outputs"
    )

    print("\nGenerated files:")
    for key, path in results.items():
        print(f"  {key}: {path}")
