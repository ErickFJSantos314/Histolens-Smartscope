"""Speech-to-Text layer for HistoLens SmartScope AI.

This module captures microphone audio in real time and transcribes it using Google Cloud
Speech-to-Text medical models. It is designed for hands-free pathology workflows where
operators should not leave the microscope controls.
"""

import argparse
import glob
import os
import wave
from typing import List, Tuple

import numpy as np
import sounddevice as sd
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import speech_v1p1beta1 as speech
from google.oauth2 import service_account


class RealtimeMedicalRecorder:
    """Captures microphone audio and triggers immediate medical transcription."""

    def __init__(
        self,
        sample_rate_hertz: int = 16000,
        channels: int = 1,
        language_code: str = "en-US",
        model: str = "medical_dictation",
    ) -> None:
        self.sample_rate_hertz = sample_rate_hertz
        self.channels = channels
        self.language_code = language_code
        self.model = model
        self._frames: List[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self.is_recording = False

    def _callback(self, indata, frames, time, status) -> None:
        if status:
            print(f"Audio stream status: {status}")
        self._frames.append(indata.copy())

    def start(self) -> None:
        """Starts microphone recording."""
        if self.is_recording:
            return

        self._frames = []
        self._stream = sd.InputStream(
            samplerate=self.sample_rate_hertz,
            channels=self.channels,
            dtype="int16",
            callback=self._callback,
        )
        self._stream.start()
        self.is_recording = True

    def stop_and_transcribe(self, output_wav_path: str = "temp_recording.wav") -> List[Tuple[str, float]]:
        """Stops recording, writes WAV file, and returns STT transcripts + confidences."""
        if not self.is_recording or self._stream is None:
            return []

        self._stream.stop()
        self._stream.close()
        self._stream = None
        self.is_recording = False

        if not self._frames:
            return []

        audio_data = np.concatenate(self._frames, axis=0)
        with wave.open(output_wav_path, "wb") as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate_hertz)
            wav_file.writeframes(audio_data.tobytes())

        return transcribe_medical(
            audio_path=output_wav_path,
            language_code=self.language_code,
            model=self.model,
            encoding="LINEAR16",
            sample_rate_hertz=self.sample_rate_hertz,
            channel_count=self.channels,
        )


def _resolve_google_credentials_path() -> str:
    """Finds credentials from env var, Downloads fallback, or local project directory."""
    candidates: List[str] = []

    env_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if env_path:
        candidates.append(os.path.expanduser(os.path.expandvars(env_path)))

    home_downloads = os.path.join(os.path.expanduser("~"), "Downloads")
    candidates.extend(sorted(glob.glob(os.path.join(home_downloads, "gen-lang-client-*.json"))))

    project_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.extend(sorted(glob.glob(os.path.join(project_dir, "*.json"))))

    for path in candidates:
        if path and os.path.exists(path):
            return path

    return ""


def _recognize_once(
    client: speech.SpeechClient,
    audio: speech.RecognitionAudio,
    config: speech.RecognitionConfig,
) -> List[Tuple[str, float]]:
    response = client.recognize(config=config, audio=audio)
    results: List[Tuple[str, float]] = []

    for result in response.results:
        alternative = result.alternatives[0]
        results.append((alternative.transcript.strip(), alternative.confidence))

    return results


def _pick_best_results(primary: List[Tuple[str, float]], fallback: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    if not primary and fallback:
        return fallback
    if not fallback:
        return primary

    primary_score = primary[0][1] if primary else 0.0
    fallback_score = fallback[0][1] if fallback else 0.0
    return fallback if fallback_score > primary_score else primary


def transcribe_medical(
    audio_path: str = "",
    gcs_uri: str = "",
    language_code: str = "en-US",
    model: str = "medical_dictation",
    encoding: str = "LINEAR16",
    sample_rate_hertz: int = 16000,
    channel_count: int = 1,
) -> List[Tuple[str, float]]:
    """Runs one-shot transcription using Google medical speech models."""
    if not audio_path and not gcs_uri:
        raise ValueError("Provide either audio_path or gcs_uri")

    credentials_path = _resolve_google_credentials_path()
    if credentials_path:
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        client = speech.SpeechClient(credentials=credentials)
    else:
        try:
            client = speech.SpeechClient()
        except DefaultCredentialsError as exc:
            raise RuntimeError(
                "Google credentials not found. Set GOOGLE_APPLICATION_CREDENTIALS or place "
                "a service-account JSON file in ~/Downloads using pattern gen-lang-client-*.json"
            ) from exc

    if gcs_uri:
        audio = speech.RecognitionAudio(uri=gcs_uri)
    else:
        with open(audio_path, "rb") as audio_file:
            content = audio_file.read()
        audio = speech.RecognitionAudio(content=content)

    forced_language_code = "en-US"
    phrase_hints = [
        "carcinoma",
        "neoplasia",
        "melanoma",
        "mitosis",
        "histopathology",
        "diagnosis",
    ]

    config = speech.RecognitionConfig(
        encoding=getattr(speech.RecognitionConfig.AudioEncoding, encoding),
        sample_rate_hertz=sample_rate_hertz,
        language_code=forced_language_code,
        model=model,
        audio_channel_count=channel_count,
        enable_word_time_offsets=True,
        use_enhanced=True,
        speech_contexts=[speech.SpeechContext(phrases=phrase_hints, boost=15.0)],
    )
    primary_results = _recognize_once(client, audio, config)

    fallback_config = speech.RecognitionConfig(
        encoding=getattr(speech.RecognitionConfig.AudioEncoding, encoding),
        sample_rate_hertz=sample_rate_hertz,
        language_code=forced_language_code,
        model="latest_short",
        audio_channel_count=channel_count,
        enable_word_time_offsets=True,
        use_enhanced=True,
        speech_contexts=[speech.SpeechContext(phrases=phrase_hints, boost=20.0)],
    )
    fallback_results = _recognize_once(client, audio, fallback_config)

    return _pick_best_results(primary_results, fallback_results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Medical Speech-to-Text (Google Cloud)")
    parser.add_argument("--audio", default="", help="Local audio file path")
    parser.add_argument("--gcs-uri", default="", help="GCS URI (gs://bucket/file)")
    parser.add_argument("--language", default="en-US", help="BCP-47 language code")
    parser.add_argument(
        "--model",
        default="medical_dictation",
        choices=["medical_dictation", "medical_conversation"],
        help="Google medical STT model",
    )
    parser.add_argument("--encoding", default="LINEAR16", help="Audio encoding")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate in Hz")
    parser.add_argument("--channels", type=int, default=1, help="Audio channel count")
    parser.add_argument(
        "--record-realtime",
        action="store_true",
        help="Interactive mode: press Enter to start and Enter again to stop",
    )
    args = parser.parse_args()

    if args.record_realtime:
        recorder = RealtimeMedicalRecorder(
            sample_rate_hertz=args.sample_rate,
            channels=args.channels,
            language_code=args.language,
            model=args.model,
        )
        print("Press Enter to start recording...")
        input()
        recorder.start()
        print("Recording... Press Enter to stop and transcribe.")
        input()
        results = recorder.stop_and_transcribe()
        for transcript, confidence in results:
            print(f"{confidence:.2f}: {transcript}")
        return

    if not args.audio and not args.gcs_uri:
        parser.error("Provide --audio or --gcs-uri (or use --record-realtime)")

    results = transcribe_medical(
        audio_path=args.audio,
        gcs_uri=args.gcs_uri,
        language_code=args.language,
        model=args.model,
        encoding=args.encoding,
        sample_rate_hertz=args.sample_rate,
        channel_count=args.channels,
    )

    for transcript, confidence in results:
        print(f"{confidence:.2f}: {transcript}")


if __name__ == "__main__":
    main()
