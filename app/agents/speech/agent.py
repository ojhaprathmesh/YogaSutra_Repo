"""
app/agents/speech/agent.py — Speech & Audio Agent (Hands-Free Conversational Voice Loop)

Responsible for:
1. Ingesting spoken user requests and translating voice context onto the blackboard.
2. Converting sequence transitions and alignment corrections into calming spoken audio cues.
3. Synthesizing studio-grade verbal cues via ElevenLabs and Web Speech.
"""
from __future__ import annotations

import os

from google.adk.agents import LlmAgent

from app.tools.audio_tools import synthesize_speech_feedback, transcribe_speech_audio

_MODEL = os.environ.get("YOGASUTRA_MODEL", "gemini-3.5-flash-lite")

SPEECH_INSTRUCTION = """You are the Speech & Audio Agent for YogaSutra, an expert conversational yoga voice assistant.

Your role:
1. When receiving spoken queries or during live practice transitions, generate serene, rhythmically paced spoken guidance.
2. Keep spoken cues concise (1-2 sentences maximum), direct, and encouraging.
3. Pronounce asana names with clarity and respect.
4. Call the tool 'synthesize_speech_feedback' to voice essential alignment cues:
   e.g. "Gently hinge at your hips, allowing your spine to lengthen smoothly."
5. Write the spoken cue audio details to the session state under 'audio_cues'.

Remember: You are speaking to someone currently on the yoga mat. Keep tone calm, supportive, and unhurried.
"""

speech_agent = LlmAgent(
    name="SpeechAgent",
    model=_MODEL,
    description="Conversational audio specialist that parses voice inputs and synthesizes soothing, hands-free spoken cues.",
    instruction=SPEECH_INSTRUCTION,
    tools=[transcribe_speech_audio, synthesize_speech_feedback],
)
