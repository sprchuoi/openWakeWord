# ESP32 Voice Detection - Quick Start Guide

Get your custom wake word running on ESP32 in under 10 minutes!

## Prerequisites

```bash
pip install openwakeword sounddevice soundfile
```

## Option 1: Zero Training (Fastest)

Use energy-based detection without any training:

```bash
cd examples
python train_custom_voice_esp32.py \
    --wake-word "hey assistant" \
    --model-type placeholder \
    --mode deploy \
    --output-dir my_esp32
```

**Result**: Ready-to-use ESP32 code in `my_esp32/` directory!

## Option 2: Train Your Voice (Best Accuracy)

### Step 1: Collect Samples (2 minutes)

```bash
python train_custom_voice_esp32.py \
    --wake-word "hey assistant" \
    --mode collect \
    --num-samples 30
```

Follow the prompts to record your voice 30 times.

### Step 2: Train Model (1 minute)

```bash
python train_custom_voice_esp32.py \
    --wake-word "hey assistant" \
    --mode train \
    --model-type placeholder
```

### Step 3: Deploy (Instant)

```bash
python train_custom_voice_esp32.py \
    --wake-word "hey assistant" \
    --mode deploy \
    --output-dir my_esp32
```

## Upload to ESP32

### Arduino IDE

1. Copy files:
```bash
cp my_esp32/hey_assistant_esp32/i2s_config.h ~/Arduino/MyProject/
```

2. Create Arduino sketch:
```cpp
#include "i2s_config.h"

void setup() {
    Serial.begin(115200);
    setup_i2s();
}

void loop() {
    int16_t buffer[BUFFER_SIZE];
    capture_audio(buffer, BUFFER_SIZE);
    
    // Your detection logic here
    float score = detect_wake_word(buffer);
    
    if (score > 0.7) {
        Serial.println("Wake word detected!");
    }
}
```

3. Upload to ESP32!

### MicroPython

1. Upload file:
```bash
ampy --port /dev/ttyUSB0 put my_esp32/hey_assistant_esp32/i2s_config.py
```

2. Create main.py:
```python
from i2s_config import capture_audio

while True:
    samples = capture_audio()
    score = detect_wake_word(samples)
    
    if score > 0.7:
        print("Wake word detected!")
```

## Hardware Wiring

```
INMP441 → ESP32
─────────────────
VDD    → 3.3V
GND    → GND  
SD     → GPIO 32
WS     → GPIO 15
SCK    → GPIO 14
L/R    → GND
```

## Interactive Notebook

For interactive training:

```bash
jupyter notebook notebooks/esp32_custom_voice_training.ipynb
```

## Test Detection

Run examples to see it in action:

```bash
python examples/esp32_examples.py
```

## Troubleshooting

**No audio?** Check wiring and I2S pins  
**False positives?** Increase threshold (0.5 → 0.7)  
**Missed detections?** Decrease threshold (0.7 → 0.5)

## Full Documentation

See `examples/README_ESP32.md` for complete guide with:
- Detailed architecture explanation
- Performance optimization tips
- Hardware selection guide
- Advanced customization

## Smart Home Integration

Example with MQTT:

```python
import paho.mqtt.client as mqtt

client = mqtt.Client()
client.connect("mqtt_broker", 1883)

def on_detection():
    client.publish("home/voice/detected", "wake_word")
    # Trigger your smart home actions
```

## Need Help?

- Check examples: `examples/esp32_examples.py`
- Read full guide: `examples/README_ESP32.md`
- Open issue on GitHub

---

**That's it!** You now have custom wake word detection running on ESP32 🎉
