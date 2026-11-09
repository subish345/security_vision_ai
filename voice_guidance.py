# ==============================================================================
# Voice Guidance Manager (Extracted from Demo 11 / Orbiter Core Assistance)
# Local, offline Text-To-Speech with priority queuing, cooldowns & deduplication
# ==============================================================================

import time
import queue
import logging
import threading
from typing import Optional, Dict

logger = logging.getLogger("VoiceGuidance")


class AudioPriority:
    CRITICAL = 1
    WARNING = 2
    GUIDANCE = 3
    INFO = 4


SEVERITY_TO_AUDIO_PRIORITY = {
    "CRITICAL": AudioPriority.CRITICAL,
    "WARNING": AudioPriority.WARNING,
    "NOTICE": AudioPriority.GUIDANCE,
    "INFO": AudioPriority.INFO,
}

COOLDOWNS = {
    AudioPriority.CRITICAL: 2.0,
    AudioPriority.WARNING: 4.0,
    AudioPriority.GUIDANCE: 3.5,
    AudioPriority.INFO: 8.0,
}


class VoiceGuidanceManager:
    """
    Offline Text-To-Speech guidance system extracted from Demo 11 / Orbiter.
    Operates on a dedicated worker thread with a priority queue and cooldown enforcement.
    """

    def __init__(self, speech_rate: int = 165, volume: float = 0.9, enabled: bool = True):
        self.enabled = enabled
        self.speech_rate = speech_rate
        self.volume = volume
        self.queue = queue.PriorityQueue()
        self.last_spoken_time: Dict[int, float] = {}
        self.phrase_cache: Dict[str, float] = {}  # phrase -> timestamp
        self.running = True
        self._engine = None
        self._init_tts_engine()

        # Start background worker thread
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()

    def _init_tts_engine(self):
        if not self.enabled:
            return
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self.speech_rate)
            self._engine.setProperty("volume", self.volume)
            logger.info("[VOICE] Offline TTS Engine initialized successfully.")
        except Exception as e:
            logger.warning(f"[VOICE] Headless Fallback: TTS audio engine unavailable ({e}). Voice will log only.")
            self._engine = None

    def speak(self, text: str, severity: str = "INFO", phrase_key: Optional[str] = None):
        """
        Enqueues a speech message with priority and cooldown validation.
        """
        if not self.enabled:
            return

        priority = SEVERITY_TO_AUDIO_PRIORITY.get(severity, AudioPriority.INFO)
        now = time.time()

        # Check priority cooldown
        last_t = self.last_spoken_time.get(priority, 0.0)
        cooldown = COOLDOWNS.get(priority, 3.0)
        if (now - last_t) < cooldown:
            return  # Dropped due to priority cooldown

        # Check deduplication cache
        dedup_key = phrase_key or text
        last_phrase_t = self.phrase_cache.get(dedup_key, 0.0)
        if (now - last_phrase_t) < 6.0:
            return  # Dropped duplicate phrase within 6s

        self.last_spoken_time[priority] = now
        self.phrase_cache[dedup_key] = now

        # Add to priority queue: tuple of (priority, timestamp, text)
        self.queue.put((priority, now, text))

    def _process_queue(self):
        while self.running:
            try:
                priority, ts, text = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                if self._engine:
                    self._engine.say(text)
                    self._engine.runAndWait()
                else:
                    logger.info(f"[VOICE SPOKEN]: \"{text}\"")
            except Exception as e:
                logger.debug(f"[VOICE ERROR] Failed speech generation: {e}")
            finally:
                self.queue.task_done()

    def stop(self):
        self.running = False


# Singleton instance
voice_system = VoiceGuidanceManager(enabled=True)

