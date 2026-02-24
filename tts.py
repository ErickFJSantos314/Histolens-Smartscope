"""Text-to-Speech layer for HistoLens SmartScope AI.

This module converts orchestrator output into short, audible responses for hands-free
pathology assistance.
"""

import argparse
import os
import tempfile
import winsound

from google.cloud import texttospeech


def _truncate_text_for_tts(text: str, max_bytes: int = 4500) -> str:
    """Caps output size for stable TTS synthesis and playback latency."""
    clean = " ".join(text.split()).strip()
    if len(clean.encode("utf-8")) <= max_bytes:
        return clean

    truncated = clean.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")
    last_break = max(truncated.rfind(". "), truncated.rfind("? "), truncated.rfind("! "))
    if last_break > 100:
        truncated = truncated[: last_break + 1]
    return truncated.strip()


def synthesize_speech(
    text: str,
    output_path: str,
    language_code: str = "en-US",
    voice_name: str = "",
    ssml_gender: str = "NEUTRAL",
    audio_encoding: str = "LINEAR16",
) -> None:
    """Calls Google Cloud TTS and writes synthesized audio to disk."""
    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice_kwargs = {
        "language_code": language_code,
        "ssml_gender": getattr(texttospeech.SsmlVoiceGender, ssml_gender),
    }
    if voice_name:
        voice_kwargs["name"] = voice_name
    voice = texttospeech.VoiceSelectionParams(**voice_kwargs)

    audio_config = texttospeech.AudioConfig(
        audio_encoding=getattr(texttospeech.AudioEncoding, audio_encoding)
    )

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config,
    )

    with open(output_path, "wb") as output_file:
        output_file.write(response.audio_content)


def speak_text(
    text: str,
    language_code: str = "en-US",
    voice_name: str = "",
    ssml_gender: str = "NEUTRAL",
) -> str:
    """Synthesizes and plays short local WAV responses, returning output path."""
    if not text.strip():
        return ""

    text_to_speak = _truncate_text_for_tts(text)
    output_path = os.path.join(tempfile.gettempdir(), "histolens_tts_response.wav")

    synthesize_speech(
        text=text_to_speak,
        output_path=output_path,
        language_code=language_code,
        voice_name=voice_name,
        ssml_gender=ssml_gender,
        audio_encoding="LINEAR16",
    )
    winsound.PlaySound(output_path, winsound.SND_FILENAME)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Google Cloud Text-to-Speech")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--out", required=True, help="Output audio file path")
    parser.add_argument("--language", default="en-US", help="BCP-47 language code")
    parser.add_argument("--voice", default="", help="Voice name (optional)")
    parser.add_argument(
        "--gender",
        default="NEUTRAL",
        choices=["NEUTRAL", "FEMALE", "MALE"],
        help="Voice gender",
    )
    parser.add_argument(
        "--encoding",
        default="LINEAR16",
        choices=["MP3", "LINEAR16", "OGG_OPUS"],
        help="Output audio encoding",
    )
    args = parser.parse_args()

    synthesize_speech(
        text=args.text,
        output_path=args.out,
        language_code=args.language,
        voice_name=args.voice,
        ssml_gender=args.gender,
        audio_encoding=args.encoding,
    )


if __name__ == "__main__":
    main()
