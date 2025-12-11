#!/bin/bash
# Setup script for ESP32 Voice Detection environment

echo "Setting up ESP32 Voice Detection environment..."

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "✓ Conda found - creating conda environment"
    conda env create -f environment_esp32.yml
    echo ""
    echo "Environment created! Activate with:"
    echo "  conda activate openwakeword-esp32"
    echo ""
    echo "Then run:"
    echo "  python examples/train_custom_voice_esp32.py --wake-word 'hey assistant' --model-type placeholder"
else
    echo "⚠ Conda not found - creating Python virtual environment instead"
    
    # Create virtual environment
    python3 -m venv venv_esp32
    
    # Activate and install dependencies
    source venv_esp32/bin/activate
    pip install --upgrade pip
    pip install numpy>=1.21.0
    pip install scipy>=1.7.0
    pip install matplotlib>=3.4.0
    pip install jupyter>=1.0.0
    pip install sounddevice>=0.4.5
    pip install soundfile>=0.11.0
    pip install tqdm>=4.62.0
    pip install pyyaml>=6.0
    pip install scikit-learn>=1.0.0
    pip install joblib>=1.1.0
    pip install requests>=2.26.0
    pip install paho-mqtt>=1.6.0
    pip install ai-edge-litert>=1.0.0 || echo "⚠ ai-edge-litert not available, TFLite features may be limited"
    
    echo ""
    echo "✓ Virtual environment created! Activate with:"
    echo "  source venv_esp32/bin/activate"
    echo ""
    echo "Then run:"
    echo "  python examples/train_custom_voice_esp32.py --wake-word 'hey assistant' --model-type placeholder"
fi

echo ""
echo "Quick start guide: cat QUICKSTART_ESP32.md"
echo "Full documentation: cat examples/README_ESP32.md"
