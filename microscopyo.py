"""
HistoLens SmartScope AI - Local Digital Twin Microscope Client.

This module simulates a microscope viewport over a WSI and executes the
hands-free loop: Speech-to-Text -> Maestro Routing -> Vision/Reasoning -> Text-to-Speech.
"""

import os
import queue
import threading
import tempfile
from dotenv import load_dotenv
if hasattr(os, 'add_dll_directory'):
    import openslide_bin
    dll_path = os.path.dirname(openslide_bin.__file__)
    os.add_dll_directory(dll_path)
from openslide import OpenSlide
import cv2
import numpy as np
from stt import RealtimeMedicalRecorder
from maestro import HistoLensOrchestrator
from tts import speak_text
from enum import Enum

load_dotenv()

# Runtime state definitions used by the simulator UI.

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

# Digital slide source for local simulation.

slide_path = os.getenv("HISTOLENS_SVS_PATH", r"archives\your-slide-file.svs")
if not os.path.exists(slide_path):
    raise FileNotFoundError(
        "SVS file not found. Set HISTOLENS_SVS_PATH (or edit slide_path) with the full path to a .svs file. "
        "Example: HISTOLENS_SVS_PATH=archives\\my_slide.svs"
    )
slide = OpenSlide(slide_path)

# Global viewport state (represents the current microscope field).

total_width, total_height = slide.dimensions
view_x = total_width // 2
view_y = total_height // 2
current_level = slide.level_count - 1

# Viewport size (simulated microscope image window).
viewport_width, viewport_height = 960, 720

# Full canvas size (viewport + telemetry overlays).
canvas_width, canvas_height = 1280, 800
border_width = (canvas_width - viewport_width) // 2
border_height = (canvas_height - viewport_height) // 2

transcript_text = ""
assistant_text = ""
current_state = HistoLensState.IDLE
status_text = "Ready"
stt_result_queue: queue.Queue = queue.Queue()
stt_worker: threading.Thread | None = None
ai_result_queue: queue.Queue = queue.Queue()
ai_worker: threading.Thread | None = None

orchestrator: HistoLensOrchestrator | None = None

voice_recorder = RealtimeMedicalRecorder(
    sample_rate_hertz=16000,
    channels=1,
    language_code="en-US",
    model="medical_conversation",
)


def transcribe_in_background() -> None:
    try:
        results = voice_recorder.stop_and_transcribe("temp_recording.wav")
        stt_result_queue.put(("ok", results))
    except Exception as e:
        stt_result_queue.put(("erro", str(e)))


def capture_current_viewport() -> str:
    screenshot_path = os.path.join(tempfile.gettempdir(), "histolens_current_view.png")
    image = slide.read_region((view_x, view_y), current_level, (viewport_width, viewport_height)).convert("RGB")
    image.save(screenshot_path)
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
    try:
        global orchestrator
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
    except Exception as e:
        ai_result_queue.put(("erro", str(e)))

def draw_panel(img, x, y, width, height, text, bg_color=(240, 240, 240), text_color=(0, 0, 0), padding=10):
    """Draws one telemetry panel on top of the microscope frame."""
    cv2.rectangle(img, (x, y), (x + width, y + height), bg_color, -1)
    cv2.rectangle(img, (x, y), (x + width, y + height), (100, 100, 100), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = x + padding
    text_y = y + (height + text_size[1]) // 2
    cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, thickness)

# Main event loop.

while True:
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
                print(f"Transcript: {transcript_text}")
                if not transcript_text:
                    print("Empty transcript. No request sent to AI.")
                    current_state = HistoLensState.IDLE
                    status_text = "Ready"
                elif ai_worker is None:
                    screenshot_path = capture_current_viewport()
                    current_state = HistoLensState.PROCESSING
                    status_text = "AI Routing..."
                    ai_worker = threading.Thread(
                        target=process_ai_in_background,
                        args=(transcript_text, screenshot_path),
                        daemon=True,
                    )
                    ai_worker.start()
                else:
                    print("AI is still processing a previous request.")
                    status_text = "AI Processing..."
            else:
                transcript_text = ""
                print("No speech detected.")
                status_text = "Ready"
        else:
            transcript_text = ""
            print(f"STT error: {payload}")
            status_text = "STT Error"

    while not ai_result_queue.empty():
        result_type, payload = ai_result_queue.get_nowait()
        current_state = HistoLensState.IDLE
        if result_type == "ok":
            route = payload.get("route")
            final_response = payload.get("final_response", "")
            route_name = route.value if route else "unknown"
            assistant_text = final_response
            print(f"Selected route: {route_name}")
            print(f"AI response: {final_response}")
            if payload.get("tts_status") == "error":
                print(f"TTS warning: {payload.get('tts_error')}")
                status_text = "TTS Error"
            else:
                status_text = "Ready"
        else:
            print(f"AI error: {payload}")
            status_text = "AI Error"

    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    
    microscope_pil = slide.read_region((view_x, view_y), current_level, (viewport_width, viewport_height))
    microscope_raw = np.array(microscope_pil)
    microscope_cv = cv2.cvtColor(microscope_raw, cv2.COLOR_RGBA2BGR)
    
    canvas[border_height:border_height + viewport_height, border_width:border_width + viewport_width] = microscope_cv
    
    info_text = f"X: {view_x:6d} | Y: {view_y:6d} | Zoom: {current_level}"
    draw_panel(canvas, 15, 15, 320, 50, info_text, bg_color=(220, 240, 255), text_color=(0, 0, 0))
    
    status_color = STATE_COLORS[current_state]
    status_box_x = canvas_width - 150
    status_box_y = 15
    cv2.rectangle(canvas, (status_box_x, status_box_y), (status_box_x + 135, status_box_y + 50), status_color, -1)
    cv2.rectangle(canvas, (status_box_x, status_box_y), (status_box_x + 135, status_box_y + 50), (0, 0, 0), 2)
    cv2.putText(canvas, "HistoLens:", (status_box_x + 8, status_box_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(canvas, status_text, (status_box_x + 8, status_box_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    if transcript_text:
        transcricao_display = f"STT: {transcript_text[:100]}"
    else:
        transcricao_display = "STT: (waiting for input...)"
    draw_panel(canvas, 15, canvas_height - 70, canvas_width - 30, 55, transcricao_display, bg_color=(240, 255, 240), text_color=(0, 100, 0))

    if assistant_text:
        resposta_display = f"AI: {assistant_text[:100]}"
    else:
        resposta_display = "AI: (waiting for response...)"
    draw_panel(canvas, 15, canvas_height - 130, canvas_width - 30, 50, resposta_display, bg_color=(235, 245, 255), text_color=(20, 20, 120))
    
    audio_status = "REC ●" if voice_recorder.is_recording else "○ LISTEN"
    draw_panel(canvas, canvas_width - 150, canvas_height - 70, 135, 55, audio_status, bg_color=(255, 240, 240), text_color=(0, 0, 0))

    cv2.namedWindow("HistoLens - SmartScope AI", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("HistoLens - SmartScope AI", canvas)

    tecla = cv2.waitKey(1) & 0xFF
    
    movement_step = 350 * (2 ** current_level)

    if tecla == ord('d'): 
        view_x += movement_step
        current_state = HistoLensState.IDLE
        status_text = "Navigating"
    if tecla == ord('a'): 
        view_x -= movement_step
        current_state = HistoLensState.IDLE
        status_text = "Navigating"
    if tecla == ord('s'): 
        view_y += movement_step
        current_state = HistoLensState.IDLE
        status_text = "Navigating"
    if tecla == ord('w'): 
        view_y -= movement_step
        current_state = HistoLensState.IDLE
        status_text = "Navigating"

    if tecla == ord('e'):
        if current_level > 0:
            current_level -= 1
            current_state = HistoLensState.IDLE
            status_text = "Zoom In"
            print(f"Zoom In: Level {current_level}")
            
    if tecla == ord('q'):
        if current_level < (slide.level_count - 1):
            current_level += 1
            current_state = HistoLensState.IDLE
            status_text = "Zoom Out"
            print(f"Zoom Out: Level {current_level}")

    if tecla == ord('r'):
        if not voice_recorder.is_recording:
            if stt_worker is not None and stt_worker.is_alive():
                print("STT is still processing. Wait before recording again.")
                current_state = HistoLensState.PROCESSING
                status_text = "Processing..."
                continue
            if ai_worker is not None and ai_worker.is_alive():
                print("AI is still processing the previous response. Wait for completion.")
                current_state = HistoLensState.PROCESSING
                status_text = "AI Processing..."
                continue
            voice_recorder.start()
            current_state = HistoLensState.LISTENING
            status_text = "Listening..."
            print("Recording audio... press 'r' again to stop.")
        else:
            current_state = HistoLensState.PROCESSING
            status_text = "Processing..."
            stt_worker = threading.Thread(target=transcribe_in_background, daemon=True)
            stt_worker.start()

    view_x = max(0, min(view_x, total_width - viewport_width))
    view_y = max(0, min(view_y, total_height - viewport_height))

    if tecla == ord('x'):
        break

cv2.destroyAllWindows()