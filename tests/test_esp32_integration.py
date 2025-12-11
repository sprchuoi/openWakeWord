#!/usr/bin/env python3
"""
Test script to verify ESP32 integration implementation.
Runs basic tests on all ModelLoader implementations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

# Import directly from modules to avoid dependency issues
import openwakeword.model_loader as ml
import openwakeword.esp32_integration as esp32

ModelLoaderFactory = ml.ModelLoaderFactory
PlaceholderModelLoader = ml.PlaceholderModelLoader
EdgeImpulseModelLoader = ml.EdgeImpulseModelLoader
CustomModelLoader = ml.CustomModelLoader

ESP32AudioConfig = esp32.ESP32AudioConfig
I2SAudioCapture = esp32.I2SAudioCapture
ESP32WakeWordDetector = esp32.ESP32WakeWordDetector


def test_placeholder_loader():
    """Test PlaceholderModelLoader."""
    print("\n" + "="*60)
    print("Testing PlaceholderModelLoader")
    print("="*60)
    
    config = {
        'energy_threshold': 0.02,
        'zcr_threshold': 0.1,
        'window_size': 1280
    }
    
    loader = ModelLoaderFactory.create_loader(
        loader_type='placeholder',
        model_path='test_placeholder.json',
        config=config
    )
    
    # Test loading
    assert loader.load() == True, "Failed to load placeholder model"
    print("✓ Model loaded")
    
    # Test input/output shapes
    assert loader.get_input_shape() == (1280,), "Incorrect input shape"
    assert loader.get_output_shape() == (1,), "Incorrect output shape"
    print(f"✓ Input shape: {loader.get_input_shape()}")
    print(f"✓ Output shape: {loader.get_output_shape()}")
    
    # Test prediction
    test_audio = np.random.randn(1280).astype(np.float32) * 0.1
    prediction = loader.predict(test_audio)
    assert prediction.shape == (1,), "Incorrect prediction shape"
    assert 0 <= prediction[0] <= 1, "Prediction out of range"
    print(f"✓ Prediction: {prediction[0]:.3f}")
    
    # Test save config
    loader.save_config('test_placeholder_saved.json')
    assert os.path.exists('test_placeholder_saved.json'), "Config not saved"
    print("✓ Config saved")
    
    # Cleanup
    if os.path.exists('test_placeholder_saved.json'):
        os.remove('test_placeholder_saved.json')
    
    print("\n✓ PlaceholderModelLoader: ALL TESTS PASSED")


def test_custom_loader():
    """Test CustomModelLoader extensibility."""
    print("\n" + "="*60)
    print("Testing CustomModelLoader")
    print("="*60)
    
    class TestCustomLoader(CustomModelLoader):
        def load(self):
            self.input_shape = (16, 96)
            self.output_shape = (1,)
            return True
        
        def predict(self, input_data):
            return np.array([0.5])
    
    # Register
    ModelLoaderFactory.register_loader('test_custom', TestCustomLoader)
    print("✓ Custom loader registered")
    
    # Create instance
    loader = ModelLoaderFactory.create_loader(
        loader_type='test_custom',
        model_path='test.bin',
        config={'format': 'test'}
    )
    
    # Test
    assert loader.load() == True, "Failed to load"
    print("✓ Model loaded")
    
    test_input = np.random.randn(16, 96).astype(np.float32)
    prediction = loader.predict(test_input)
    assert prediction[0] == 0.5, "Incorrect prediction"
    print(f"✓ Prediction: {prediction[0]:.3f}")
    
    print("\n✓ CustomModelLoader: ALL TESTS PASSED")


def test_factory():
    """Test ModelLoaderFactory."""
    print("\n" + "="*60)
    print("Testing ModelLoaderFactory")
    print("="*60)
    
    available = ModelLoaderFactory.get_available_loaders()
    print(f"Available loaders: {available}")
    
    assert 'placeholder' in available, "Placeholder not available"
    assert 'edge_impulse' in available, "EdgeImpulse not available"
    assert 'custom' in available, "Custom not available"
    print("✓ All expected loaders available")
    
    # Test creation
    loader = ModelLoaderFactory.create_loader('placeholder', 'test.json')
    assert isinstance(loader, PlaceholderModelLoader), "Wrong loader type"
    print("✓ Factory creates correct loader type")
    
    print("\n✓ ModelLoaderFactory: ALL TESTS PASSED")


def test_esp32_audio_config():
    """Test ESP32AudioConfig."""
    print("\n" + "="*60)
    print("Testing ESP32AudioConfig")
    print("="*60)
    
    config = ESP32AudioConfig(
        sample_rate=16000,
        bits_per_sample=16,
        channels=1,
        buffer_size=1280
    )
    
    assert config.sample_rate == 16000, "Wrong sample rate"
    assert config.buffer_size == 1280, "Wrong buffer size"
    assert config.buffer_duration_ms == 80.0, "Wrong buffer duration"
    print(f"✓ Sample rate: {config.sample_rate} Hz")
    print(f"✓ Buffer duration: {config.buffer_duration_ms} ms")
    
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict), "to_dict failed"
    print("✓ Configuration export works")
    
    print("\n✓ ESP32AudioConfig: ALL TESTS PASSED")


def test_i2s_audio_capture():
    """Test I2SAudioCapture."""
    print("\n" + "="*60)
    print("Testing I2SAudioCapture")
    print("="*60)
    
    config = ESP32AudioConfig()
    capture = I2SAudioCapture(config)
    
    # Test Arduino config generation
    arduino_code = capture.generate_esp32_config()
    assert 'i2s_config_t' in arduino_code, "Arduino config missing"
    assert 'SAMPLE_RATE' in arduino_code, "Sample rate missing"
    print("✓ Arduino I2S configuration generated")
    
    # Test MicroPython config generation
    micropython_code = capture.generate_micropython_config()
    assert 'I2S' in micropython_code, "MicroPython I2S missing"
    assert 'capture_audio' in micropython_code, "Capture function missing"
    print("✓ MicroPython I2S configuration generated")
    
    # Test simulation
    audio = capture.simulate_capture(duration_ms=80)
    assert audio.shape[0] == 1280, "Wrong audio length"
    assert audio.dtype == np.int16, "Wrong audio dtype"
    print(f"✓ Audio simulation: {audio.shape} samples")
    
    print("\n✓ I2SAudioCapture: ALL TESTS PASSED")


def test_esp32_detector():
    """Test ESP32WakeWordDetector."""
    print("\n" + "="*60)
    print("Testing ESP32WakeWordDetector")
    print("="*60)
    
    # Create detector
    loader = ModelLoaderFactory.create_loader('placeholder', 'test.json')
    audio_config = ESP32AudioConfig()
    
    detector = ESP32WakeWordDetector(
        model_loader=loader,
        audio_config=audio_config,
        detection_threshold=0.5,
        smoothing_window=5
    )
    print("✓ Detector initialized")
    
    # Test detection
    audio = detector.audio_capture.simulate_capture()
    detected, score = detector.detect(audio)
    
    assert isinstance(detected, (bool, np.bool_)), "Wrong detection type"
    assert isinstance(score, (float, np.floating)), "Wrong score type"
    assert 0 <= score <= 1, "Score out of range"
    print(f"✓ Detection: {detected}, Score: {score:.3f}")
    
    # Test multiple frames
    for i in range(5):
        audio = detector.audio_capture.simulate_capture()
        detected, score = detector.detect(audio)
    print("✓ Multiple frame processing works")
    
    # Test benchmarking
    metrics = detector.benchmark_performance(num_iterations=10)
    assert 'mean_inference_time_ms' in metrics, "Missing metric"
    assert metrics['mean_inference_time_ms'] > 0, "Invalid timing"
    print(f"✓ Benchmark: {metrics['mean_inference_time_ms']:.2f} ms/frame")
    
    # Test deployment package generation
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        detector.generate_deployment_package(tmpdir)
        
        assert os.path.exists(os.path.join(tmpdir, 'i2s_config.h')), "Arduino config missing"
        assert os.path.exists(os.path.join(tmpdir, 'i2s_config.py')), "MicroPython config missing"
        assert os.path.exists(os.path.join(tmpdir, 'detector_config.json')), "Detector config missing"
        assert os.path.exists(os.path.join(tmpdir, 'README.md')), "README missing"
        print("✓ Deployment package generated")
    
    print("\n✓ ESP32WakeWordDetector: ALL TESTS PASSED")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print(" ESP32 Integration Test Suite")
    print("="*70)
    
    try:
        test_factory()
        test_placeholder_loader()
        test_custom_loader()
        test_esp32_audio_config()
        test_i2s_audio_capture()
        test_esp32_detector()
        
        print("\n" + "="*70)
        print(" ✅ ALL TESTS PASSED!")
        print("="*70)
        print("\nYour ESP32 integration is working correctly!")
        print("Next steps:")
        print("  1. Run examples: python examples/esp32_examples.py")
        print("  2. Train your model: python examples/train_custom_voice_esp32.py")
        print("  3. Read the guide: cat examples/README_ESP32.md")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
