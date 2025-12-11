# ESP32 Wake Word Detector: hey assistant

## Hardware Setup

### Wiring (INMP441 Microphone)
```
Microphone    ESP32
─────────────────────
VDD      →    3.3V
GND      →    GND
SD       →    GPIO 32
WS       →    GPIO 15
SCK      →    GPIO 14
L/R      →    GND
```

## Arduino Example

```cpp
#include "i2s_config.h"

void setup() {
    Serial.begin(115200);
    setup_i2s();
    Serial.println("Wake word detector ready!");
}

void loop() {
    int16_t buffer[BUFFER_SIZE];
    capture_audio(buffer, BUFFER_SIZE);
    
    float score = detect_wake_word(buffer, BUFFER_SIZE);
    
    if (score > 0.7) {
        Serial.println("Wake word detected!");
        // Your action here
    }
    
    delay(10);
}
```

## MicroPython Example

```python
from i2s_config import capture_audio, detect_wake_word

print("Wake word detector ready!")

while True:
    samples = capture_audio()
    score = detect_wake_word(samples)
    
    if score > 0.7:
        print("Wake word detected!")
        # Your action here
```

## Configuration

- Model: Energy-based (PlaceholderModelLoader)
- Sample Rate: 16kHz
- Detection: RMS energy threshold
- No ML dependencies required
- Ultra-fast inference (<1ms)

## Tuning

Adjust detection sensitivity in code:
- Higher threshold (0.02 → 0.03) = Less sensitive
- Lower threshold (0.02 → 0.01) = More sensitive

## Next Steps

1. Copy i2s_config.h to Arduino project
2. Wire up ESP32 + microphone
3. Upload code
4. Speak "hey assistant" to test
5. Adjust threshold as needed

Generated with openWakeWord ESP32 Integration
