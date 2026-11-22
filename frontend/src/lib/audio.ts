import { resolveApiUrl } from "./api";

let activeAudio: HTMLAudioElement | null = null;

export function stopSpokenAudio(): void {
  if (activeAudio) {
    activeAudio.pause();
    activeAudio.currentTime = 0;
    activeAudio = null;
  }
  if (typeof window !== "undefined" && "speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }
}

export function speakWithWebSpeech(
  text: string,
  onEnd?: () => void
): boolean {
  if (typeof window === "undefined" || !("speechSynthesis" in window) || !text.trim()) {
    onEnd?.();
    return false;
  }
  try {
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.96;
    utterance.pitch = 1.0;
    utterance.onend = () => onEnd?.();
    utterance.onerror = () => onEnd?.();
    window.speechSynthesis.speak(utterance);
    return true;
  } catch {
    onEnd?.();
    return false;
  }
}

export async function playSpokenCue(options: {
  text: string;
  audioUrl?: string | null;
  onStart?: (source: "elevenlabs" | "webspeech") => void;
  onEnd?: () => void;
}): Promise<"elevenlabs" | "webspeech" | "none"> {
  const { text, audioUrl, onStart, onEnd } = options;
  stopSpokenAudio();

  const resolvedUrl = resolveApiUrl(audioUrl);
  if (resolvedUrl && typeof window !== "undefined") {
    try {
      const audio = new Audio(resolvedUrl);
      activeAudio = audio;
      audio.onended = () => {
        if (activeAudio === audio) activeAudio = null;
        onEnd?.();
      };
      audio.onerror = () => {
        if (activeAudio === audio) activeAudio = null;
        onStart?.("webspeech");
        speakWithWebSpeech(text, onEnd);
      };
      await audio.play();
      onStart?.("elevenlabs");
      return "elevenlabs";
    } catch {
      // Browser autoplay or network fallback to WebSpeech
      activeAudio = null;
    }
  }

  if (text.trim()) {
    onStart?.("webspeech");
    const ok = speakWithWebSpeech(text, onEnd);
    return ok ? "webspeech" : "none";
  }

  onEnd?.();
  return "none";
}
