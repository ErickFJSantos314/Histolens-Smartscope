"""
HistoLens Maestro Router and Orchestrator.

The Maestro layer is the cognitive controller that decides how each user request should
be processed in a latency-aware and clinically safe way.

Pipeline strategy:
1) Triage intent with a lightweight Gemini model.
2) Dispatch to direct chat, remote vision analysis, technical guidance, or deep diagnostic synthesis.
3) Normalize final output for spoken delivery (TTS-safe, concise, no markdown artifacts).
"""

import json
import os
import re
from enum import Enum
from typing import Callable, Optional, Tuple

import google.generativeai as genai
from dotenv import load_dotenv

from medgemma import MedGemmaModel, create_medgemma_engine


load_dotenv()


class RouteType(Enum):
    """Supported inference routes in the layered HistoLens architecture."""

    BASIC_CHAT = "basic_chat"
    IMAGE_INTERACTION = "image_interaction"
    TECHNICAL_DOUBT = "technical_doubt"
    HEAVY_DIAGNOSTIC = "heavy_diagnostic"


class MaestroRouter:
    """
    Intent classifier for pathologist queries.

    This component does not produce final diagnosis. It only decides the most appropriate
    computational path based on user intent and risk level.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        on_status_change: Optional[Callable[[str, str], None]] = None,
    ):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY is missing. Configure it in environment variables or .env file."
            )

        self.on_status_change = on_status_change or self._default_status_handler
        genai.configure(api_key=self.api_key)
        self.model_name = self._resolve_model_name()
        self.model = genai.GenerativeModel(self.model_name)

        self.system_prompt = """You are the HistoLens triage engine for WSI histopathology workflows.
Your task is strict intent routing.

Classify each input into exactly one route:

1) BASIC_CHAT
- Greetings, social interactions, generic assistant questions.

2) IMAGE_INTERACTION
- Visual/morphological description of the current microscope field without advanced diagnosis.
- Any request to analyze/describe the current image should map here.

3) TECHNICAL_DOUBT
- Theoretical references, protocols, guidelines, classifications, ICD questions.
- Current image is not required.

4) HEAVY_DIAGNOSTIC
- Complex diagnostic support requiring morphology + differential reasoning.

Return JSON only:
{
  "route": "BASIC_CHAT" | "IMAGE_INTERACTION" | "TECHNICAL_DOUBT" | "HEAVY_DIAGNOSTIC",
  "confidence": 0.98,
  "reasoning": "short rationale"
}
"""

    def _default_status_handler(self, status_name: str, status_text: str) -> None:
        print(f"[Maestro] {status_name}: {status_text}")

    def _resolve_model_name(self) -> str:
        preferred_models = [
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ]

        for model_name in preferred_models:
            try:
                probe = genai.GenerativeModel(model_name)
                probe.generate_content("ok")
                self.on_status_change("MODEL", f"Using Gemini model: {model_name}")
                return model_name
            except Exception:
                continue

        for model_info in genai.list_models():
            methods = getattr(model_info, "supported_generation_methods", [])
            if "generateContent" not in methods:
                continue
            raw_name = getattr(model_info, "name", "")
            model_name = raw_name.replace("models/", "")
            if model_name.startswith("gemini"):
                self.on_status_change("MODEL", f"Fallback Gemini model selected: {model_name}")
                return model_name

        raise RuntimeError("No compatible Gemini model found for generateContent")

    def _generate_content(self, prompt: str):
        try:
            return self.model.generate_content(prompt)
        except Exception as exc:
            message = str(exc).lower()
            if "404" in message or "not found" in message or "not supported" in message:
                self.model_name = self._resolve_model_name()
                self.model = genai.GenerativeModel(self.model_name)
                return self.model.generate_content(prompt)
            raise

    @staticmethod
    def _strip_emojis(text: str) -> str:
        emoji_pattern = re.compile(
            "["
            "\U0001F300-\U0001FAFF"
            "\U00002700-\U000027BF"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        return emoji_pattern.sub("", text).strip()

    def classify_query(self, query: str) -> Tuple[RouteType, float, str]:
        self.on_status_change("MAESTRO", "Analyzing user intent...")

        quick_query = query.lower()
        if any(token in quick_query for token in ["image", "slide", "wsi", "histology", "histopathology"]):
            if any(token in quick_query for token in ["analyze", "analyse", "describe", "look", "see"]):
                return RouteType.IMAGE_INTERACTION, 0.99, "Explicit image-analysis request"

        try:
            response = self._generate_content(f"{self.system_prompt}\n\nPathologist input: {query}")
            response_text = response.text.strip()

            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]

            result = json.loads(response_text)

            route_map = {
                "BASIC_CHAT": RouteType.BASIC_CHAT,
                "IMAGE_INTERACTION": RouteType.IMAGE_INTERACTION,
                "TECHNICAL_DOUBT": RouteType.TECHNICAL_DOUBT,
                "HEAVY_DIAGNOSTIC": RouteType.HEAVY_DIAGNOSTIC,
            }
            route = route_map.get(result.get("route", ""), RouteType.HEAVY_DIAGNOSTIC)
            confidence = result.get("confidence", 0.0)
            reasoning = result.get("reasoning", "")

            self.on_status_change("ROUTE", f"{route.value} (confidence: {confidence:.0%})")
            return route, confidence, reasoning

        except Exception as exc:
            self.on_status_change("ERROR", f"Intent classification failed: {exc}")
            return RouteType.HEAVY_DIAGNOSTIC, 0.0, f"Classifier fallback due to error: {exc}"

    def ask_gemini(self, prompt: str) -> str:
        response = self._generate_content(prompt)
        return self._strip_emojis(response.text.strip())


class HistoLensOrchestrator:
    """Runs end-to-end decision and response synthesis for all HistoLens routes."""

    def __init__(
        self,
        google_api_key: Optional[str] = None,
        hf_token: Optional[str] = None,
        on_status_change: Optional[Callable[[str, str], None]] = None,
    ):
        self.on_status_change = on_status_change or self._default_status_handler
        self.router = MaestroRouter(api_key=google_api_key, on_status_change=self.on_status_change)
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self._vision_engine = None

    def _default_status_handler(self, status_name: str, status_text: str) -> None:
        print(f"[Orchestrator] {status_name}: {status_text}")

    @staticmethod
    def _normalize_tts_text(text: str, max_chars: int = 700) -> str:
        if not text:
            return ""

        text = text.replace("**", "")
        text = re.sub(r"^\s*#{1,6}\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+\)\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
        text = text.replace("*", "")
        text = text.replace(":", ".")
        text = text.replace(" -- ", " ")
        text = text.replace(" - ", " ")
        text = re.sub(r"[;,_-]{2,}", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) > max_chars:
            cutoff = text.rfind(".", 0, max_chars)
            text = text[:max_chars].rstrip() if cutoff == -1 else text[: cutoff + 1].strip()

        return text

    def _get_vision_engine(self):
        if self._vision_engine is None:
            self.on_status_change("MODEL", "Connecting to MedGemma 4B remote API...")
            self._vision_engine = create_medgemma_engine(
                MedGemmaModel.VISION_4B,
                self.hf_token,
                on_status_change=self.on_status_change,
            )
        return self._vision_engine

    def _run_vision_4b(self, query: str, screenshot_path: Optional[str]) -> str:
        if not screenshot_path:
            return "No screenshot is available for visual analysis."

        engine = self._get_vision_engine()
        return engine.infer(
            screenshot_path,
            prompt=(
                "You are observing a histopathology image. Describe only visible evidence using plain sentences. "
                "Include cellularity, nuclei, cytoplasm, architecture, staining, and artifacts. "
                "Do not provide recommendations. Do not use bullet points, numbering, colons, or markdown. "
                f"User context: {query}"
            ),
        )

    @staticmethod
    def _is_medgemma_error(text: str) -> bool:
        lowered = text.lower()
        return any(
            token in lowered
            for token in [
                "api error",
                "api timeout",
                "api connection error",
                "model load error",
                "import error",
                "not a valid model identifier",
                "medgemma failure",
            ]
        )

    def handle_query(
        self,
        query: str,
        screenshot_path: Optional[str] = None,
        route_override: Optional[Tuple[RouteType, float, str]] = None,
    ) -> dict:
        route, confidence, reasoning = route_override or self.router.classify_query(query)
        self.on_status_change("ROUTE", f"{route.value} ({confidence:.0%})")

        result = {
            "route": route,
            "confidence": confidence,
            "reasoning": reasoning,
            "query": query,
            "screenshot_path": screenshot_path,
            "gemini_response": None,
            "vision_response": None,
            "theory_response": None,
            "final_response": "",
        }

        if route == RouteType.BASIC_CHAT:
            self.on_status_change("FLOW", "Basic: direct Gemini response")
            final = self.router.ask_gemini(
                "You are HistoLens, a conversational assistant for histopathology workflows. "
                f"User message: '{query}'\n\n"
                "Rules: concise, professional, same language as user, no emojis, single short paragraph."
            )
            result["gemini_response"] = final
            result["final_response"] = self._normalize_tts_text(final, max_chars=400)
            return result

        if route == RouteType.IMAGE_INTERACTION:
            self.on_status_change("FLOW", "Image interaction: Gemini planning + MedGemma vision")
            vision_brief = self.router.ask_gemini(
                "Convert the pathologist request into a strict visual extraction instruction for MedGemma. "
                "Focus only on morphology. No diagnosis. No emojis."
                f"\n\nOriginal question: '{query}'"
            )
            result["gemini_response"] = vision_brief

            try:
                vision_response = self._run_vision_4b(vision_brief, screenshot_path)
            except Exception as exc:
                vision_response = f"MedGemma 4B error: {exc}"
            result["vision_response"] = vision_response

            if self._is_medgemma_error(vision_response):
                final = self.router.ask_gemini(
                    "Visual model is unavailable. Explain limitation and ask user to validate endpoint access. "
                    "Same language as user. No emojis."
                    f"\n\nUser question: {query}"
                )
                result["final_response"] = self._normalize_tts_text(final, max_chars=400)
                return result

            final = self.router.ask_gemini(
                "You are assisting a pathologist. Use only the morphology below and answer directly. "
                "No invented diagnosis unless explicitly requested. Single paragraph under 5 sentences."
                f"\n\nMorphology: {vision_response}\n\nUser question: {query}"
            )
            result["final_response"] = self._normalize_tts_text(final, max_chars=700)
            return result

        if route == RouteType.TECHNICAL_DOUBT:
            self.on_status_change("FLOW", "Technical: Gemini knowledge response")
            final = self.router.ask_gemini(
                "You are HistoLens Medical Board. Provide technically precise pathology guidance. "
                "Same language as user. No emojis. Single paragraph under 8 sentences."
                f"\n\nQuestion: '{query}'"
            )
            result["gemini_response"] = final
            result["final_response"] = self._normalize_tts_text(final, max_chars=900)
            return result

        self.on_status_change("FLOW", "Heavy diagnostic: MedGemma vision + Gemini synthesis")

        visual_plan = self.router.ask_gemini(
            "Extract a strict visual analysis plan for a pathology vision model. "
            "List what to inspect in nuclei, architecture, and atypia, but output as plain sentences."
            f"\n\nRequest: '{query}'"
        )
        result["gemini_response"] = visual_plan

        try:
            visual_report = self._run_vision_4b(visual_plan, screenshot_path)
        except Exception as exc:
            visual_report = f"MedGemma 4B API error: {exc}"
        result["vision_response"] = visual_report

        if self._is_medgemma_error(visual_report):
            fallback = self.router.ask_gemini(
                "Visual analysis is unavailable. Provide cautious guidance, clearly state limitations. "
                "Same language as user. No emojis."
                f"\n\nQuestion: {query}"
            )
            result["final_response"] = self._normalize_tts_text(fallback, max_chars=600)
            return result

        board_prompt = f"""You are the Chief Pathologist in HistoLens Medical Board.

Morphological analysis from MedGemma:
{visual_report}

Clinical question:
{query}

Requirements:
- Synthesize visual findings with clinical context.
- Propose differential diagnostic hypotheses with confidence framing.
- Keep output as one paragraph.
- No bullet points, no numbering, no markdown, no asterisks.
- Keep under 140 words.
"""

        final_diagnosis = self.router.ask_gemini(board_prompt)
        result["theory_response"] = final_diagnosis
        result["final_response"] = self._normalize_tts_text(final_diagnosis, max_chars=900)
        return result

    async def process_query_async(self, query: str, screenshot_path: Optional[str] = None) -> dict:
        return self.handle_query(query=query, screenshot_path=screenshot_path)


if __name__ == "__main__":
    router = MaestroRouter()
    samples = [
        "hello",
        "Describe this slide morphology",
        "What ICD code applies to basal cell carcinoma?",
        "Analyze this tissue and provide differential diagnosis",
    ]

    for query in samples:
        print(f"\n>>> {query}")
        route, confidence, rationale = router.classify_query(query)
        print(f"Route: {route.value} | Confidence: {confidence:.0%}")
        print(f"Rationale: {rationale}")
