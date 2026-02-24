"""
MedGemma Remote Vision Client for HistoLens SmartScope AI.

This module implements the cloud-edge contract between the local microscope client and
remote MedGemma inference infrastructure (Kaggle/Colab + Ngrok tunnel).

Design goals:
- Keep the local workstation lightweight.
- Offload GPU-heavy visual morphology extraction to remote infrastructure.
- Return compact, clinically useful text to downstream synthesis and TTS.
"""

import base64
import io
import logging
import os
from enum import Enum
from typing import Callable, Optional

import requests


_base_dir = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
LOG_PATH = os.path.join(_base_dir, "medgemma_debug.log")
logging.basicConfig(
    filename=LOG_PATH,
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


class MedGemmaModel(Enum):
    """Available remote model endpoints used by the orchestrator."""

    VISION_4B = "medgemma_4b"


class MedGemma4BVisionAPI:
    """
    Lightweight HTTP client for MedGemma 1.5 4B visual inference.

    Runtime contract:
    - Input: image + prompt
    - Transport: JSON over HTTPS
    - Output: morphological description returned by remote GPU server
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_token: Optional[str] = None,
        on_status_change: Optional[Callable[[str, str], None]] = None,
    ):
        self.api_url = api_url or os.getenv("REMOTE_VISION_API_URL") or os.getenv("COLAB_API_URL")
        if not self.api_url:
            raise ValueError(
                "REMOTE_VISION_API_URL is not configured. Set REMOTE_VISION_API_URL to your active "
                "Ngrok endpoint (for example: https://xxxx.ngrok-free.dev/analyze)."
            )

        self.api_token = api_token or os.getenv("REMOTE_VISION_API_TOKEN") or os.getenv("COLAB_API_TOKEN")
        self.on_status_change = on_status_change or self._default_status_handler
        self.model_type = MedGemmaModel.VISION_4B

        self._notify(f"API endpoint: {self.api_url}")

    def _default_status_handler(self, status_name: str, status_text: str) -> None:
        print(f"[{status_name}] {status_text}")

    def _notify(self, message: str) -> None:
        self.on_status_change(self.model_type.value, message)

    def _image_to_base64(self, image_source) -> str:
        """
        Converts an image path or PIL image into Base64 JPEG payload.
        """
        try:
            from PIL import Image

            image = Image.open(image_source) if isinstance(image_source, str) else image_source

            max_side = int(os.getenv("MEDGEMMA_VISION_MAX_SIDE", "512"))
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            logging.info("4B API: image resized to %s", image.size)

            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            image_bytes = buffer.getvalue()
            encoded = base64.b64encode(image_bytes).decode("utf-8")

            logging.info("4B API: image converted to Base64 (%d bytes)", len(image_bytes))
            return encoded
        except Exception as exc:
            logging.exception("4B API: image conversion failed")
            raise ValueError(f"Failed to process image input: {exc}") from exc

    def infer(
        self,
        image_path,
        prompt: str = "Describe the histological morphology visible in this image.",
    ) -> str:
        """
        Sends an image to remote MedGemma server and returns morphology text.
        """
        self._notify("Uploading image to remote vision API...")
        logging.info("4B API: inference start")

        try:
            image_base64 = self._image_to_base64(image_path)

            payload = {
                "image": image_base64,
                "prompt": prompt,
                "max_tokens": int(os.getenv("MEDGEMMA_VISION_MAX_TOKENS", "128")),
            }

            headers = {"Content-Type": "application/json"}
            if self.api_token:
                headers["Authorization"] = f"Bearer {self.api_token}"

            self._notify("Waiting for remote GPU response...")
            logging.info("4B API: POST %s", self.api_url)

            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=120,
            )
            response.raise_for_status()

            result = response.json()
            if result.get("status") == "error" or "error" in result:
                error_msg = result.get("message", result.get("error", "Unknown remote error"))
                logging.error("4B API: remote error: %s", error_msg)
                self._notify(f"Remote vision API error: {error_msg}")
                return f"API Error: {error_msg}"

            text_response = result.get("response", result.get("text", ""))
            logging.info("4B API: response received (%d chars)", len(text_response))
            self._notify("Visual analysis completed.")
            return text_response

        except requests.Timeout:
            error_msg = "Remote vision API timed out after 120 seconds"
            logging.error("4B API: %s", error_msg)
            self._notify(error_msg)
            return f"API Timeout: {error_msg}"

        except requests.RequestException as exc:
            error_msg = f"Remote connection failure: {exc}"
            logging.exception("4B API: request failed")
            self._notify(error_msg)
            return f"API Connection Error: {error_msg}"

        except Exception as exc:
            logging.exception("4B API: unexpected runtime error")
            self._notify(f"Unexpected MedGemma client error: {exc}")
            raise


def create_medgemma_engine(
    model_type: MedGemmaModel,
    hf_token: Optional[str] = None,
    on_status_change: Optional[Callable] = None,
):
    """Factory for MedGemma engine instances used by Maestro."""
    if model_type == MedGemmaModel.VISION_4B:
        return MedGemma4BVisionAPI(on_status_change=on_status_change)
    raise ValueError(f"Unsupported model type: {model_type}. Only VISION_4B is available.")


if __name__ == "__main__":
    print("HistoLens MedGemma Remote Client")
    print(
        "- REMOTE_VISION_API_URL: "
        f"{os.getenv('REMOTE_VISION_API_URL', os.getenv('COLAB_API_URL', 'NOT CONFIGURED'))}"
    )
    print(
        "- REMOTE_VISION_API_TOKEN: "
        f"{'Configured' if (os.getenv('REMOTE_VISION_API_TOKEN') or os.getenv('COLAB_API_TOKEN')) else 'Not configured'}"
    )

    if os.getenv("REMOTE_VISION_API_URL") or os.getenv("COLAB_API_URL"):
        print("\nClient is ready.")
        print("Example:")
        print("  vision_api = MedGemma4BVisionAPI()")
        print("  result = vision_api.infer('path/to/slide.jpg', 'Describe tissue morphology')")
    else:
        print("\nSet REMOTE_VISION_API_URL before running inference.")
