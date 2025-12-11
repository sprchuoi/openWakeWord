
# ESP32 I2S Configuration for Wake Word: hey assistant
# Energy-based detection

from machine import I2S, Pin
import array
import math

SAMPLE_RATE = 16000
BUFFER_SIZE = 1280

# I2S pins
SCK_PIN = 14
WS_PIN = 15
SD_PIN = 32

audio_in = I2S(
    0,
    sck=Pin(SCK_PIN),
    ws=Pin(WS_PIN),
    sd=Pin(SD_PIN),
    mode=I2S.RX,
    bits=16,
    format=I2S.MONO,
    rate=SAMPLE_RATE,
    ibuf=BUFFER_SIZE * 2
)

def capture_audio():
    audio_buffer = bytearray(BUFFER_SIZE * 2)
    audio_in.readinto(audio_buffer)
    return array.array('h', audio_buffer)

def detect_wake_word(samples):
    # Calculate RMS energy
    sum_sq = sum((s / 32768.0) ** 2 for s in samples)
    energy = math.sqrt(sum_sq / len(samples))
    
    # Detection
    threshold = 0.02
    return energy / threshold if energy > threshold else 0.0
