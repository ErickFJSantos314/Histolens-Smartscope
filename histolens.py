"""
HistoLens SmartScope AI - Real-time microscope camera client.

This module is the original application concept: it captures frames from a live
camera feed (USB/c-mount), runs the hands-free AI loop, and responds by voice.
"""

import os
import queue
import tempfile
import threading
from enum import Enum

import cv2
from dotenv import load_dotenv

from maestro import HistoLensOrchestrator
from stt import RealtimeMedicalRecorder
from tts import speak_text

load_dotenv()


class HistoLensState(Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    SPEAKING = "SPEAKING"
    RECORDING = "RECORDING"


STATE_COLORS = {
    HistoLensState.IDLE: (255, 255, 255),
    HistoLensState.LISTENING: (255, 200, 100),
    HistoLensState.PROCESSING: (255, 100, 50),
    HistoLensState.SPEAKING: (0, 255, 200),
    HistoLensState.RECORDING: (0, 0, 255),
}

transcript_text = ""
assistant_text = ""
status_text = "Ready"
current_state = HistoLensState.IDLE

stt_result_queue: queue.Queue = queue.Queue()
ai_result_queue: queue.Queue = queue.Queue()
stt_worker: threading.Thread | None = None
ai_worker: threading.Thread | None = None

orchestrator: HistoLensOrchestrator | None = None
current_frame = None

voice_recorder = RealtimeMedicalRecorder(
    sample_rate_hertz=16000,
    channels=1,
    language_code="en-US",
    model="medical_conversation",
)


def draw_panel(img, x, y, width, height, text, bg_color=(240, 240, 240), text_color=(0, 0, 0), padding=10):
    cv2.rectangle(img, (x, y), (x + width, y + height), bg_color, -1)
    cv2.rectangle(img, (x, y), (x + width, y + height), (100, 100, 100), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = x + padding
    text_y = y + (height + text_size[1]) // 2
    cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, thickness)


def transcribe_in_background() -> None:
    try:
        results = voice_recorder.stop_and_transcribe("temp_recording.wav")
        stt_result_queue.put(("ok", results))
    except Exception as exc:
        stt_result_queue.put(("erro", str(exc)))


def capture_current_frame_snapshot() -> str:
    if current_frame is None:
        raise RuntimeError("No camera frame available.")
    screenshot_path = os.path.join(tempfile.gettempdir(), "histolens_current_frame.png")
    cv2.imwrite(screenshot_path, current_frame)
    return screenshot_path


def select_tts_text(text: str) -> str:
    if not text:
        return ""

    marker = "Fallback Gemini:"
    if marker in text:
        text = text.split(marker, 1)[1].strip()

    lowered = text.lower()
    if lowered.startswith("medgemma failure") or lowered.startswith("error "):
        return ""

    return text.strip()


def process_ai_in_background(query_text: str, screenshot_path: str) -> None:
    global orchestrator
    try:
        if orchestrator is None:
            orchestrator = HistoLensOrchestrator(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                hf_token=os.getenv("HF_TOKEN"),
            )

        route_info = orchestrator.router.classify_query(query_text)

        if route_info[0].value in ["image_interaction", "technical_doubt", "heavy_diagnostic"]:
            try:
                speak_text("This may take a moment.")
            except Exception:
                pass

        result = orchestrator.handle_query(
            query_text,
            screenshot_path=screenshot_path,
            route_override=route_info,
        )

        final_text = result.get("final_response", "").strip()
        speech_text = select_tts_text(final_text)
        if speech_text:
            try:
                speak_text(speech_text)
                result["tts_status"] = "ok"
            except Exception as tts_error:
                result["tts_status"] = "error"
                result["tts_error"] = str(tts_error)

        ai_result_queue.put(("ok", result))
    except Exception as exc:
        ai_result_queue.put(("erro", str(exc)))


def main() -> None:
    global stt_worker, ai_worker
    global transcript_text, assistant_text, current_state, status_text, current_frame

    camera_index = int(os.getenv("HISTOLENS_CAMERA_INDEX", "0"))
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}. Set HISTOLENS_CAMERA_INDEX to a valid device id."
        )

    frame_w = int(os.getenv("HISTOLENS_FRAME_WIDTH", "1280"))
    frame_h = int(os.getenv("HISTOLENS_FRAME_HEIGHT", "720"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)

    window_title = "HistoLens - Real-time SmartScope AI"
    cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                status_text = "Camera Error"
                current_state = HistoLensState.IDLE
                break

            current_frame = frame.copy()
            canvas = frame.copy()
            canvas_h, canvas_w = canvas.shape[:2]

            if stt_worker is not None and not stt_worker.is_alive():
                stt_worker = None
            if ai_worker is not None and not ai_worker.is_alive():
                ai_worker = None

            while not stt_result_queue.empty():
                result_type, payload = stt_result_queue.get_nowait()
                current_state = HistoLensState.IDLE
                if result_type == "ok":
                    if payload:
                        transcript_text = " ".join(item[0] for item in payload).strip()
                        if not transcript_text:
                            status_text = "Ready"
                        elif ai_worker is None:
                            snapshot_path = capture_current_frame_snapshot()
                            current_state = HistoLensState.PROCESSING
                            status_text = "AI Routing..."
                            ai_worker = threading.Thread(
                                target=process_ai_in_background,
                                args=(transcript_text, snapshot_path),
                                daemon=True,
                            )
                            ai_worker.start()
                        else:
                            status_text = "AI Processing..."
                    else:
                        transcript_text = ""
                        status_text = "Ready"
                else:
                    transcript_text = ""
                    status_text = "STT Error"

            while not ai_result_queue.empty():
                result_type, payload = ai_result_queue.get_nowait()
                current_state = HistoLensState.IDLE
                if result_type == "ok":
                    assistant_text = payload.get("final_response", "")
                    if payload.get("tts_status") == "error":
                        status_text = "TTS Error"
                    else:
                        status_text = "Ready"
                else:
                    status_text = "AI Error"

            status_color = STATE_COLORS[current_state]
            status_box_x = canvas_w - 170
            status_box_y = 15
            cv2.rectangle(canvas, (status_box_x, status_box_y), (status_box_x + 155, status_box_y + 50), status_color, -1)
            cv2.rectangle(canvas, (status_box_x, status_box_y), (status_box_x + 155, status_box_y + 50), (0, 0, 0), 2)
            cv2.putText(canvas, "HistoLens:", (status_box_x + 8, status_box_y + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(canvas, status_text, (status_box_x + 8, status_box_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

            camera_info = f"Camera: {camera_index} | {canvas_w}x{canvas_h}"
            draw_panel(canvas, 15, 15, 320, 50, camera_info, bg_color=(220, 240, 255), text_color=(0, 0, 0))

            if transcript_text:
                stt_display = f"STT: {transcript_text[:120]}"
            else:
                stt_display = "STT: (waiting for input...)"
            draw_panel(canvas, 15, canvas_h - 70, canvas_w - 200, 55, stt_display, bg_color=(240, 255, 240), text_color=(0, 100, 0))

            if assistant_text:
                ai_display = f"AI: {assistant_text[:120]}"
            else:
                ai_display = "AI: (waiting for response...)"
            draw_panel(canvas, 15, canvas_h - 130, canvas_w - 200, 50, ai_display, bg_color=(235, 245, 255), text_color=(20, 20, 120))

            audio_status = "REC ●" if voice_recorder.is_recording else "○ LISTEN"
            draw_panel(canvas, canvas_w - 170, canvas_h - 70, 155, 55, audio_status, bg_color=(255, 240, 240), text_color=(0, 0, 0))

            draw_panel(canvas, canvas_w - 170, canvas_h - 130, 155, 50, "R: record | X: exit", bg_color=(245, 245, 245), text_color=(0, 0, 0))

            cv2.imshow(window_title, canvas)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"):
                if not voice_recorder.is_recording:
                    if stt_worker is not None and stt_worker.is_alive():
                        current_state = HistoLensState.PROCESSING
                        status_text = "Processing..."
                        continue
                    if ai_worker is not None and ai_worker.is_alive():
                        current_state = HistoLensState.PROCESSING
                        status_text = "AI Processing..."
                        continue
                    voice_recorder.start()
                    current_state = HistoLensState.LISTENING
                    status_text = "Listening..."
                else:
                    current_state = HistoLensState.PROCESSING
                    status_text = "Processing..."
                    stt_worker = threading.Thread(target=transcribe_in_background, daemon=True)
                    stt_worker.start()

            if key == ord("x"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
