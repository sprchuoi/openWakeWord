# ESP32 Wake Word Detection - Complete Guide

Complete guide for implementing custom voice detection on ESP32 for smart home applications using the openWakeWord framework with ModelLoader architecture.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Model Loaders](#model-loaders)
5. [Training Custom Models](#training-custom-models)
6. [ESP32 Deployment](#esp32-deployment)
7. [Hardware Setup](#hardware-setup)
8. [Performance Optimization](#performance-optimization)
9. [Troubleshooting](#troubleshooting)
10. [Examples](#examples)

## Overview

This integration provides a complete solution for wake word detection on ESP32 microcontrollers, featuring:

- **Abstract ModelLoader interface** with factory pattern
- **Multiple inference backends**:
  - PlaceholderModelLoader: Energy-based detection (zero dependencies)
  - EdgeImpulseModelLoader: TensorFlow Lite Micro integration
  - CustomModelLoader: Extension point for proprietary formats
- **I2S audio capture** support for ESP32
- **Low-power optimization** strategies
- **Complete training pipeline** for custom wake words

## Architecture

### ModelLoader Factory Pattern

```
┌─────────────────────────────────────┐
│     ModelLoaderFactory              │
│  (Factory Pattern Implementation)   │
└──────────┬──────────────────────────┘
           │
           ├─► PlaceholderModelLoader
           │   • Energy-based detection
           │   • Zero ML dependencies
           │   • ~0.1-0.5 ms inference
           │
           ├─► EdgeImpulseModelLoader
           │   • TensorFlow Lite Micro
           │   • Edge Impulse compatible
           │   • ~5-20 ms inference
           │
           └─► CustomModelLoader
               • Extensible interface
               • Your custom format
               • Custom inference backend
```

### System Architecture

```
┌──────────────┐
│ ESP32 + I2S  │
│  Microphone  │
└──────┬───────┘
       │ Audio Stream (I2S)
       ▼
┌──────────────────────┐
│  I2SAudioCapture     │
│  • Buffer management │
│  • DMA configuration │
└──────┬───────────────┘
       │ Audio Frames
       ▼
┌──────────────────────┐
│ ESP32WakeWordDetector│
│  • Preprocessing     │
│  • Score smoothing   │
└──────┬───────────────┘
       │ Features
       ▼
┌──────────────────────┐
│   ModelLoader        │
│  • Inference         │
│  • Score output      │
└──────┬───────────────┘
       │ Detection Score
       ▼
┌──────────────────────┐
│   Smart Home         │
│   Integration        │
└──────────────────────┘
```

## Quick Start

### 1. Installation

```bash
cd openWakeWord
pip install -e .
pip install sounddevice soundfile  # For training
```

### 2. Run Examples

```bash
# View all examples
python examples/esp32_examples.py

# Test placeholder model (fastest, no training needed)
python -c "from examples.esp32_examples import example_1_placeholder_model; example_1_placeholder_model()"
```

### 3. Train Your Custom Wake Word

```bash
# Complete pipeline: collect samples, train, and deploy
python examples/train_custom_voice_esp32.py \
    --wake-word "hey assistant" \
    --mode full \
    --num-samples 50 \
    --output-dir my_wake_word

# Use placeholder model (no training, energy-based)
python examples/train_custom_voice_esp32.py \
    --wake-word "hey assistant" \
    --mode full \
    --model-type placeholder \
    --output-dir my_wake_word
```

### 4. Deploy to ESP32

The deployment package is automatically generated in the output directory with:
- I2S configuration for Arduino/ESP-IDF
- MicroPython I2S configuration
- Model file (if applicable)
- Configuration JSON
- Complete README with wiring instructions

## Model Loaders

### PlaceholderModelLoader

**Best for**: Quick prototyping, low-resource ESP32, energy-based voice detection

**Features**:
- Zero ML dependencies
- Ultra-fast inference (~0.1-0.5 ms)
- Configurable energy and zero-crossing rate thresholds
- Works immediately without training

**Usage**:

```python
from openwakeword.model_loader import ModelLoaderFactory

config = {
    'energy_threshold': 0.02,
    'zcr_threshold': 0.1,
    'window_size': 1280
}

model_loader = ModelLoaderFactory.create_loader(
    loader_type='placeholder',
    model_path='config.json',
    config=config
)
```

**Configuration Parameters**:
- `energy_threshold`: Minimum RMS energy (tune for environment noise)
- `zcr_threshold`: Zero-crossing rate threshold (voice vs. noise)
- `window_size`: Analysis window in samples
- `smoothing_window`: Temporal smoothing window

### EdgeImpulseModelLoader

**Best for**: Production deployments, custom trained models, high accuracy

**Features**:
- TensorFlow Lite Micro support
- Edge Impulse model compatibility
- Hardware acceleration (when available)
- Custom model training support

**Usage**:

```python
config = {
    'num_threads': 1,
    'use_xnnpack': True
}

model_loader = ModelLoaderFactory.create_loader(
    loader_type='edge_impulse',
    model_path='model.tflite',
    config=config
)
```

**Requirements**:
- Trained TFLite model file
- Input shape: typically (16, 96) for melspectrogram
- Output shape: (1,) for binary classification

### CustomModelLoader

**Best for**: Proprietary formats, custom inference engines

**Usage**:

```python
from openwakeword.model_loader import CustomModelLoader

class MyCustomLoader(CustomModelLoader):
    def load(self):
        # Your loading logic
        self.input_shape = (16, 96)
        self.output_shape = (1,)
        return True
    
    def predict(self, input_data):
        # Your inference logic
        return np.array([score])

# Register and use
ModelLoaderFactory.register_loader('my_format', MyCustomLoader)
loader = ModelLoaderFactory.create_loader('my_format', 'model.bin')
```

## Training Custom Models

### Method 1: Using Placeholder (No Training)

```bash
python examples/train_custom_voice_esp32.py \
    --wake-word "hey assistant" \
    --model-type placeholder \
    --output-dir deployment
```

Configures energy-based detection without requiring training data.

### Method 2: Record Your Voice

```bash
# Step 1: Collect samples
python examples/train_custom_voice_esp32.py \
    --wake-word "hey assistant" \
    --mode collect \
    --num-samples 50

# Step 2: Train model
python examples/train_custom_voice_esp32.py \
    --wake-word "hey assistant" \
    --mode train \
    --model-type dnn \
    --positive-dir voice_samples

# Step 3: Deploy
python examples/train_custom_voice_esp32.py \
    --wake-word "hey assistant" \
    --mode deploy \
    --output-dir deployment
```

### Method 3: Using Pre-recorded Data

Place your WAV files in directories:
```
positive_samples/
  hey_assistant_001.wav
  hey_assistant_002.wav
  ...

negative_samples/
  noise_001.wav
  speech_001.wav
  ...
```

Then train:
```bash
python examples/train_custom_voice_esp32.py \
    --wake-word "hey assistant" \
    --mode train \
    --positive-dir positive_samples \
    --negative-dir negative_samples \
    --model-type dnn
```

## ESP32 Deployment

### Hardware Requirements

- **ESP32 Board**: ESP32, ESP32-S2, ESP32-S3, or ESP32-C3
- **I2S Microphone**: INMP441, SPH0645, or ICS-43434
- **Power Supply**: Stable 3.3V/5V supply
- **Optional**: LED for detection indication

### Wiring Diagram

```
INMP441 Microphone    ESP32
─────────────────────────────
VDD (3.3V)      →     3.3V
GND             →     GND
SD (DOUT)       →     GPIO 32
WS (LRCLK)      →     GPIO 15
SCK (BCLK)      →     GPIO 14
L/R (GND)       →     GND
```

### Arduino IDE Setup

1. **Install ESP32 Board Support**:
   - File → Preferences → Additional Board URLs
   - Add: `https://dl.espressif.com/dl/package_esp32_index.json`
   - Tools → Board → Boards Manager → Install "ESP32"

2. **Copy Generated Files**:
   ```bash
   cp deployment/i2s_config.h ~/Arduino/WakeWordProject/
   cp deployment/model.tflite ~/Arduino/WakeWordProject/data/
   ```

3. **Arduino Sketch**:

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
    
    // Process with your model
    float score = detect_wake_word(buffer, BUFFER_SIZE);
    
    if (score > 0.7) {
        Serial.println("Wake word detected!");
        // Trigger your smart home action
    }
    
    delay(10);
}
```

### MicroPython Setup

1. **Flash MicroPython**:
   ```bash
   esptool.py --port /dev/ttyUSB0 erase_flash
   esptool.py --port /dev/ttyUSB0 write_flash -z 0x1000 esp32-micropython.bin
   ```

2. **Upload Files**:
   ```bash
   ampy --port /dev/ttyUSB0 put deployment/i2s_config.py
   ampy --port /dev/ttyUSB0 put deployment/model.tflite
   ```

3. **MicroPython Script**:

```python
from i2s_config import capture_audio
import time

print("Wake word detector ready!")

while True:
    samples = capture_audio()
    
    # Process with your model
    score = detect_wake_word(samples)
    
    if score > 0.7:
        print("Wake word detected!")
        # Trigger your smart home action
    
    time.sleep(0.01)
```

## Performance Optimization

### Model Selection Guide

| ESP32 Model | RAM | Clock | Recommended Loader | Notes |
|-------------|-----|-------|-------------------|-------|
| ESP32 Classic | 520KB | 240MHz | Placeholder | Energy-based only |
| ESP32-S2 | 320KB | 240MHz | Placeholder | Limited RAM |
| ESP32-S3 | 512KB | 240MHz | Edge Impulse | Good TFLite support |
| ESP32-S3 + PSRAM | 8MB+ | 240MHz | Edge Impulse | Best for complex models |

### Latency Optimization

```python
# Low latency configuration (40ms buffer)
config = ESP32AudioConfig(
    sample_rate=16000,
    buffer_size=640,
    dma_buffer_count=4,
    dma_buffer_size=160
)

# Balanced configuration (80ms buffer)
config = ESP32AudioConfig(
    sample_rate=16000,
    buffer_size=1280,
    dma_buffer_count=8,
    dma_buffer_size=320
)
```

### Power Optimization

1. **Use Light Sleep**: Sleep between audio buffers
2. **Reduce Sample Rate**: 8kHz for simple wake words
3. **Energy Pre-filter**: Use PlaceholderModelLoader as gatekeeper
4. **Adaptive Processing**: Only run full model on suspicious frames

### Cascade Detection Example

```python
# Stage 1: Fast energy-based pre-filter
placeholder_loader = ModelLoaderFactory.create_loader(
    'placeholder', 'config.json',
    config={'energy_threshold': 0.01}
)

# Stage 2: Accurate TFLite model
tflite_loader = ModelLoaderFactory.create_loader(
    'edge_impulse', 'model.tflite'
)

# Process audio
energy_score = placeholder_loader.predict(audio_data)

if energy_score > 0.3:  # Pre-filter threshold
    # Only run expensive model if energy detected
    final_score = tflite_loader.predict(features)
```

## Troubleshooting

### No Audio / Silence

- **Check wiring**: Verify all I2S pins are connected
- **Check power**: Ensure microphone has stable 3.3V
- **Check L/R pin**: Should be tied to GND or VCC
- **Test with serial print**: Print raw audio values

### Noisy / Distorted Audio

- **Add decoupling capacitor**: 100nF near microphone VDD
- **Check ground**: Ensure solid ground connection
- **Reduce DMA buffer size**: May help with timing issues
- **Check clock stability**: Verify ESP32 has good power

### False Positives

- **Increase threshold**: Raise detection_threshold (e.g., 0.5 → 0.7)
- **Add more smoothing**: Increase smoothing_window (e.g., 5 → 10)
- **Improve training data**: Add more negative examples
- **Adjust energy threshold**: Increase for noisy environments

### Missed Detections

- **Lower threshold**: Reduce detection_threshold (e.g., 0.7 → 0.5)
- **Check microphone placement**: Ensure clear audio path
- **Verify volume**: Speak louder or move closer
- **Retrain model**: Use more diverse positive samples

### High CPU Usage

- **Use Placeholder model**: Switch to energy-based detection
- **Increase buffer size**: Reduce processing frequency
- **Optimize model**: Use smaller TFLite model
- **Enable hardware acceleration**: Use XNNPACK if available

## Examples

### Example 1: Basic Detection

```python
from openwakeword.model_loader import ModelLoaderFactory
from openwakeword.esp32_integration import ESP32WakeWordDetector, ESP32AudioConfig

# Create detector
loader = ModelLoaderFactory.create_loader('placeholder', 'config.json')
detector = ESP32WakeWordDetector(loader)

# Simulate detection
audio = detector.audio_capture.simulate_capture()
detected, score = detector.detect(audio)
print(f"Detected: {detected}, Score: {score:.3f}")
```

### Example 2: Complete Smart Home Integration

```python
import paho.mqtt.client as mqtt

# Setup MQTT for smart home
client = mqtt.Client()
client.connect("mqtt_broker", 1883)

# Setup detector
loader = ModelLoaderFactory.create_loader('placeholder', 'config.json')
detector = ESP32WakeWordDetector(loader, detection_threshold=0.7)

def on_wake_word_detected():
    """Callback when wake word is detected."""
    print("Wake word detected! Activating smart home...")
    client.publish("home/assistant/wake", "detected")
    # Add your smart home actions here

# Main loop
while True:
    audio = capture_i2s_audio()  # Your I2S capture function
    detected, score = detector.detect(audio)
    
    if detected:
        on_wake_word_detected()
```

### Example 3: Multi-Language Support

```python
# Load multiple models for different languages
loaders = {
    'english': ModelLoaderFactory.create_loader('edge_impulse', 'model_en.tflite'),
    'spanish': ModelLoaderFactory.create_loader('edge_impulse', 'model_es.tflite'),
    'french': ModelLoaderFactory.create_loader('edge_impulse', 'model_fr.tflite'),
}

current_language = 'english'
detector = ESP32WakeWordDetector(loaders[current_language])

# Switch language dynamically
def change_language(new_language):
    global detector, current_language
    current_language = new_language
    detector = ESP32WakeWordDetector(loaders[new_language])
```

## Advanced Topics

### Custom Feature Extraction

For advanced use cases, you can implement custom feature extraction:

```python
class CustomFeatureExtractor:
    def extract_features(self, audio):
        # Compute MFCCs, spectrograms, or other features
        import librosa
        mfccs = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13)
        return mfccs.T  # Return (time, features) shape
```

### Continuous Learning

Implement online learning to improve detection over time:

```python
# Collect false positives and retrain
false_positives = []

def collect_feedback(audio, prediction):
    user_feedback = get_user_confirmation()
    if prediction and not user_feedback:
        false_positives.append(audio)
    
    if len(false_positives) > 100:
        retrain_model(false_positives)
```

## Additional Resources

- **openWakeWord Documentation**: [GitHub](https://github.com/dscripka/openWakeWord)
- **ESP32 I2S Documentation**: [Espressif](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/peripherals/i2s.html)
- **Edge Impulse**: [edgeimpulse.com](https://edgeimpulse.com)
- **TensorFlow Lite Micro**: [TensorFlow](https://www.tensorflow.org/lite/microcontrollers)

## License

This ESP32 integration follows the same Apache 2.0 license as openWakeWord.

## Contributing

Contributions are welcome! Please submit issues and pull requests on the openWakeWord GitHub repository.

## Support

For questions and support:
- Open an issue on GitHub
- Check existing documentation
- Join the community discussions
