"""
Text-to-Speech Module
Uses Bark for high-quality voice synthesis
"""
import numpy as np
from scipy.io.wavfile import write as write_wav
from pathlib import Path


class AudioGenerator:
    def __init__(self):
        self.loaded = False
        self.sample_rate = 24000  # Bark's sample rate

    def load(self):
        """Load Bark models"""
        if self.loaded:
            return

        print("Loading Bark TTS model...")
        from bark import preload_models, SAMPLE_RATE
        preload_models()
        self.sample_rate = SAMPLE_RATE
        self.loaded = True
        print("✅ Audio model loaded!")

    def generate(
        self,
        text: str,
        output_path: str = "outputs/generated_audio.wav",
        voice_preset: str = None
    ) -> str:
        """
        Generate speech from text

        Args:
            text: Text to convert to speech
            output_path: Where to save the audio
            voice_preset: Optional voice preset (e.g., "v2/en_speaker_6")

        Returns:
            Path to the generated audio file
        """
        self.load()

        from bark import generate_audio

        print(f"Generating speech for: '{text[:50]}...'")

        # Generate audio
        audio_array = generate_audio(
            text,
            history_prompt=voice_preset
        )

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save to file
        write_wav(output_path, self.sample_rate, audio_array)

        print(f"✅ Audio saved to {output_path}")
        return output_path

    def generate_with_music(
        self,
        text: str,
        output_path: str = "outputs/generated_audio.wav"
    ) -> str:
        """Generate speech with background music notation"""
        # Bark supports special tokens for music/sounds
        # ♪ for music, [laughter], [sighs], etc.
        enhanced_text = text
        return self.generate(enhanced_text, output_path)


class SimpleAudioGenerator:
    """
    Fallback TTS using edge-tts (no GPU needed, very fast)
    """
    def __init__(self, voice: str = "en-US-AriaNeural"):
        self.voice = voice

    async def generate_async(
        self,
        text: str,
        output_path: str = "outputs/generated_audio.mp3"
    ) -> str:
        """Generate speech using Edge TTS"""
        import edge_tts

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(output_path)

        print(f"✅ Audio saved to {output_path}")
        return output_path

    def generate(self, text: str, output_path: str = "outputs/generated_audio.mp3") -> str:
        """Sync wrapper for generate_async"""
        import asyncio
        return asyncio.run(self.generate_async(text, output_path))


# Standalone usage
if __name__ == "__main__":
    # Try Bark first (GPU recommended)
    try:
        generator = AudioGenerator()
        audio_path = generator.generate(
            "Hello! This is an AI generated voice. Isn't it amazing?",
            "outputs/bark_audio.wav"
        )
    except Exception as e:
        print(f"Bark failed: {e}")
        print("Falling back to Edge TTS...")

        # Fallback to Edge TTS (CPU, fast)
        # pip install edge-tts
        generator = SimpleAudioGenerator()
        audio_path = generator.generate(
            "Hello! This is an AI generated voice using Edge TTS.",
            "outputs/edge_audio.mp3"
        )
