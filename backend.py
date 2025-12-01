import os
import json
from datetime import datetime
from collections import deque, Counter

import numpy as np
import pyaudio

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "listening_data.json")

DB_FLOOR = -60.0


# ========================= JSON LOGGING =========================

def json_log_listen(username: str, mood: str, avg_db):
    try:
        avg_db_py = float(avg_db)
    except Exception:
        avg_db_py = 0.0

    entry = {
        "user": str(username),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mood": str(mood),
        "avg_db": round(avg_db_py, 2),
    }

    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = []
        except Exception:
            data = []
    else:
        data = []

    data.append(entry)

    try:
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"[JSON] Session logged -> {entry}")
    except Exception as e:
        print(f"[JSON] Failed to write log: {e}")


# ========================= AUDIO HELPERS =========================

def estimate_spectral_centroid(samples, rate):
    if len(samples) == 0:
        return 0.0
    window = np.hanning(len(samples))
    x = samples * window
    spec = np.fft.rfft(x)
    mag = np.abs(spec)
    if np.sum(mag) < 1e-8: # If the sound is too tiny to mean anything, skip the math and return zero.
        return 0.0
    freqs = np.fft.rfftfreq(len(x), d=1.0 / rate)
    centroid = np.sum(freqs * mag) / np.sum(mag)
    return float(centroid)


def classify_mood(db_val: float, energy: float, centroid: float) -> str:
    if db_val < -35.0 and energy < 0.03:
        return "silence"

    if db_val > -18 and centroid > 2200:
        return "energetic"

    if db_val < -24 and centroid < 1700:
        return "sad"

    if db_val > -28:
        return "chill"

    return "chill"


def make_amplitude_bins(samples, num_bins=64):
    if len(samples) == 0:
        return np.zeros(num_bins, dtype=np.float32)
    chunks = np.array_split(samples, num_bins)
    mags = np.array([np.mean(np.abs(c)) for c in chunks], dtype=np.float32)
    max_mag = float(np.max(mags)) if np.max(mags) > 1e-9 else 1.0
    mags /= max_mag
    return mags


# ========================= BACKEND CLASS =========================

class AudioBackend:
    def __init__(self, rate=44100, chunk=1024):
        self.rate = rate
        self.chunk = chunk

        self.pa = pyaudio.PyAudio()
        self.stream = None

        self.devices = self._list_input_devices()
        if not self.devices:
            raise RuntimeError("No input devices found.")

        self.device_idx = 0
        self.channels = self.devices[self.device_idx]["channels"]

        print("\n[AUDIO] Available input devices:")
        for i, d in enumerate(self.devices):
            print(
                f"  [{i}] {d['name']} "
                f"(dev_index={d['index']}, channels={d['channels']}, rate={d['rate']})"
            )
        print(f"[AUDIO] Starting with device [0]: {self.devices[0]['name']}\n")

        self._open_stream()

        self.alpha = 0.08
        self.ema_db = DB_FLOOR
        self.energy = 0.0
        self.centroid = 0.0
        self.samples = np.zeros(self.chunk, dtype=np.float32)

        self.mood_history = deque(maxlen=30)
        self.current_mood = "silence"

        self.mood_counter = Counter()
        self.db_sum = 0.0
        self.db_count = 0

        self.username = None

    # ---------------- device listing ----------------

    def _list_input_devices(self):
        devices = []
        try:
            count = self.pa.get_device_count()
            for i in range(count):
                info = self.pa.get_device_info_by_index(i)
                if info.get("maxInputChannels", 0) > 0:
                    devices.append(
                        {
                            "index": info["index"],
                            "name": info.get("name", f"Device {i}"),
                            "channels": max(1, info.get("maxInputChannels", 1)),
                            "rate": int(info.get("defaultSampleRate", 44100)),
                        }
                    )
        except Exception as e:
            print(f"[AUDIO] Failed to list devices: {e}")
        return devices

    # ---------------- open/cycle streams ----------------

    def _open_stream(self):
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
            self.stream = None

        attempts = 0
        total = len(self.devices)

        while attempts < total:
            dev = self.devices[self.device_idx]
            self.channels = dev["channels"]
            dev_rate = dev["rate"]

            print(
                f"[AUDIO] Opening device: {dev['name']} "
                f"(index={dev['index']}, channels={self.channels}, rate={dev_rate})"
            )

            try:
                self.stream = self.pa.open(
                    format=pyaudio.paInt16,
                    channels=self.channels,
                    rate=dev_rate,
                    input=True,
                    input_device_index=dev["index"],
                    frames_per_buffer=self.chunk,
                )
                self.rate = dev_rate
                return
            except OSError as e:
                print(f"[AUDIO] Failed: {e}")
                self.stream = None
                self.device_idx = (self.device_idx + 1) % total
                attempts += 1

        raise RuntimeError("Could not open any device.")

    def next_device(self):
        self.device_idx = (self.device_idx + 1) % len(self.devices)
        self._open_stream()

    def get_input_label(self):
        name = self.devices[self.device_idx]["name"]
        return (name[:25] + "...") if len(name) > 28 else name

    # ---------------- runtime ----------------

    def set_username(self, username: str):
        self.username = username

    def process_frame(self):
        data = self.stream.read(self.chunk, exception_on_overflow=False)
        samples_i16 = np.frombuffer(data, dtype=np.int16)

        if self.channels > 1:
            try:
                samples_i16 = (
                    samples_i16.reshape(-1, self.channels)
                    .mean(axis=1)
                    .astype(np.int16)
                )
            except:
                pass

        self.samples = samples_i16.astype(np.float32) / 32768.0

        if len(self.samples) > 0:
            rms = np.sqrt(np.mean(self.samples**2))
        else:
            rms = 0.0

        inst_db = -60.0 if rms < 1e-6 else 20.0 * np.log10(rms)

        # smooth dB with EMA
        alpha = self.alpha
        new_ema = (1 - alpha) * self.ema_db + alpha * inst_db

        # hard limit: don't let it jump too fast per frame
        max_step = 0.8  # max dB change per frame
        if new_ema > self.ema_db + max_step:
            new_ema = self.ema_db + max_step
        elif new_ema < self.ema_db - max_step:
            new_ema = self.ema_db - max_step

        self.ema_db = new_ema

        # rest stays same
        self.energy = float(np.mean(np.abs(self.samples))) if len(self.samples) else 0.0
        self.centroid = estimate_spectral_centroid(self.samples, self.rate)


        mood_now = classify_mood(self.ema_db, self.energy, self.centroid)
        self.mood_history.append(mood_now)
        self.current_mood = Counter(self.mood_history).most_common(1)[0][0]

        if self.username and self.current_mood != "silence":
            self.mood_counter[self.current_mood] += 1
            self.db_sum += float(self.ema_db)
            self.db_count += 1

    def finalize_session(self):
        if not self.username or self.db_count == 0:
            print("[JSON] Nothing to log.")
            return

        dominant_mood, _ = self.mood_counter.most_common(1)[0]
        avg_db = self.db_sum / self.db_count
        json_log_listen(self.username, dominant_mood, avg_db)

    def close(self):
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
        except:
            pass
        try:
            self.pa.terminate()
        except:
            pass
