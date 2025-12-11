# Copyright 2022 David Scripka. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Abstract ModelLoader interface with factory pattern for flexible model loading
Supports multiple inference backends including energy-based detection, 
TensorFlow Lite Micro for ESP32, and custom formats.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import numpy as np
import os
import logging


class ModelLoader(ABC):
    """
    Abstract base class for model loaders.
    Implements the factory pattern to support multiple inference backends.
    """
    
    def __init__(self, model_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model loader.
        
        Args:
            model_path: Path to the model file or configuration
            config: Optional configuration dictionary for the model
        """
        self.model_path = model_path
        self.config = config or {}
        self.model = None
        self.input_shape = None
        self.output_shape = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def load(self) -> bool:
        """
        Load the model from the specified path.
        
        Returns:
            bool: True if loading successful, False otherwise
        """
        pass
    
    @abstractmethod
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference on the input data.
        
        Args:
            input_data: Input features as numpy array
            
        Returns:
            np.ndarray: Model predictions
        """
        pass
    
    @abstractmethod
    def get_input_shape(self) -> tuple:
        """
        Get the expected input shape for the model.
        
        Returns:
            tuple: Input shape (excluding batch dimension)
        """
        pass
    
    @abstractmethod
    def get_output_shape(self) -> tuple:
        """
        Get the output shape of the model.
        
        Returns:
            tuple: Output shape (excluding batch dimension)
        """
        pass
    
    def cleanup(self):
        """
        Clean up resources used by the model loader.
        Override if specific cleanup is needed.
        """
        self.model = None


class PlaceholderModelLoader(ModelLoader):
    """
    Energy-based detection model with zero dependencies.
    Perfect for ESP32 deployment with minimal memory footprint.
    Uses simple energy thresholding and zero-crossing rate for voice activity detection.
    """
    
    def __init__(self, model_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize placeholder model loader.
        
        Config options:
            - energy_threshold: float, minimum energy threshold (default: 0.01)
            - zcr_threshold: float, zero-crossing rate threshold (default: 0.1)
            - window_size: int, analysis window size in samples (default: 400)
            - smoothing_window: int, smoothing window for energy (default: 10)
        """
        super().__init__(model_path, config)
        self.energy_threshold = self.config.get('energy_threshold', 0.01)
        self.zcr_threshold = self.config.get('zcr_threshold', 0.1)
        self.window_size = self.config.get('window_size', 400)
        self.smoothing_window = self.config.get('smoothing_window', 10)
        self.energy_history = []
        
    def load(self) -> bool:
        """Load model configuration from file if it exists."""
        if os.path.exists(self.model_path):
            try:
                import json
                with open(self.model_path, 'r') as f:
                    config = json.load(f)
                    self.energy_threshold = config.get('energy_threshold', self.energy_threshold)
                    self.zcr_threshold = config.get('zcr_threshold', self.zcr_threshold)
                    self.window_size = config.get('window_size', self.window_size)
                    self.logger.info(f"Loaded placeholder model config from {self.model_path}")
            except Exception as e:
                self.logger.warning(f"Could not load config from {self.model_path}: {e}")
        
        self.input_shape = (self.window_size,)
        self.output_shape = (1,)
        return True
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Perform energy-based detection on input audio.
        
        Args:
            input_data: Audio samples as numpy array
            
        Returns:
            np.ndarray: Detection score [0, 1]
        """
        if input_data.size == 0:
            return np.array([0.0])
        
        # Calculate energy (RMS)
        energy = np.sqrt(np.mean(input_data ** 2))
        
        # Calculate zero-crossing rate
        zcr = np.sum(np.abs(np.diff(np.sign(input_data)))) / (2 * len(input_data))
        
        # Smooth energy over time
        self.energy_history.append(energy)
        if len(self.energy_history) > self.smoothing_window:
            self.energy_history.pop(0)
        
        smoothed_energy = np.mean(self.energy_history)
        
        # Combine metrics for detection score
        energy_score = min(1.0, smoothed_energy / self.energy_threshold)
        zcr_score = min(1.0, zcr / self.zcr_threshold)
        
        # Weighted combination
        detection_score = 0.7 * energy_score + 0.3 * zcr_score
        
        return np.array([min(1.0, detection_score)])
    
    def get_input_shape(self) -> tuple:
        return self.input_shape
    
    def get_output_shape(self) -> tuple:
        return self.output_shape
    
    def save_config(self, save_path: Optional[str] = None):
        """Save the current configuration to a JSON file."""
        import json
        save_path = save_path or self.model_path
        config = {
            'energy_threshold': self.energy_threshold,
            'zcr_threshold': self.zcr_threshold,
            'window_size': self.window_size,
            'smoothing_window': self.smoothing_window
        }
        with open(save_path, 'w') as f:
            json.dump(config, f, indent=2)
        self.logger.info(f"Saved placeholder model config to {save_path}")


class EdgeImpulseModelLoader(ModelLoader):
    """
    TensorFlow Lite Micro integration scaffold for ESP32 deployment.
    Compatible with Edge Impulse exported models.
    """
    
    def __init__(self, model_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Edge Impulse model loader.
        
        Config options:
            - use_xnnpack: bool, enable XNNPACK delegate for optimization (default: True)
            - num_threads: int, number of threads for inference (default: 1)
        """
        super().__init__(model_path, config)
        self.use_xnnpack = self.config.get('use_xnnpack', True)
        self.num_threads = self.config.get('num_threads', 1)
        self.interpreter = None
        
    def load(self) -> bool:
        """Load TensorFlow Lite model for Edge Impulse inference."""
        try:
            # Try ai_edge_litert first (modern TFLite runtime)
            try:
                import ai_edge_litert.interpreter as tflite
            except ImportError:
                # Fallback to older tensorflow.lite
                try:
                    import tensorflow.lite as tflite
                except ImportError:
                    self.logger.error("Neither ai_edge_litert nor tensorflow.lite available")
                    return False
            
            # Load model
            self.interpreter = tflite.Interpreter(
                model_path=self.model_path,
                num_threads=self.num_threads
            )
            self.interpreter.allocate_tensors()
            
            # Get input/output details
            input_details = self.interpreter.get_input_details()[0]
            output_details = self.interpreter.get_output_details()[0]
            
            self.input_shape = tuple(input_details['shape'][1:])  # Remove batch dimension
            self.output_shape = tuple(output_details['shape'][1:])
            
            self.logger.info(f"Loaded Edge Impulse model from {self.model_path}")
            self.logger.info(f"Input shape: {self.input_shape}, Output shape: {self.output_shape}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load Edge Impulse model: {e}")
            return False
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run TFLite inference on input features.
        
        Args:
            input_data: Input features matching the model's input shape
            
        Returns:
            np.ndarray: Model predictions
        """
        if self.interpreter is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Ensure correct shape (add batch dimension if needed)
        if len(input_data.shape) == len(self.input_shape):
            input_data = np.expand_dims(input_data, axis=0)
        
        # Set input tensor
        input_details = self.interpreter.get_input_details()[0]
        self.interpreter.set_tensor(input_details['index'], input_data.astype(input_details['dtype']))
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output_details = self.interpreter.get_output_details()[0]
        output_data = self.interpreter.get_tensor(output_details['index'])
        
        return output_data.squeeze()
    
    def get_input_shape(self) -> tuple:
        return self.input_shape
    
    def get_output_shape(self) -> tuple:
        return self.output_shape
    
    def export_for_esp32(self, output_path: str, include_metadata: bool = True):
        """
        Export model in format optimized for ESP32 deployment.
        
        Args:
            output_path: Path to save the exported model
            include_metadata: Whether to include metadata about input/output shapes
        """
        import shutil
        
        # Copy the TFLite model
        shutil.copy2(self.model_path, output_path)
        
        if include_metadata:
            # Save metadata
            metadata = {
                'input_shape': list(self.input_shape),
                'output_shape': list(self.output_shape),
                'num_threads': self.num_threads,
                'use_xnnpack': self.use_xnnpack
            }
            
            import json
            metadata_path = output_path.replace('.tflite', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Exported model to {output_path} with metadata")


class CustomModelLoader(ModelLoader):
    """
    Extension point for proprietary or custom model formats.
    Provides a template for implementing custom inference backends.
    """
    
    def __init__(self, model_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize custom model loader.
        
        Config should include:
            - format: str, the custom format identifier
            - Any format-specific configuration
        """
        super().__init__(model_path, config)
        self.format = self.config.get('format', 'unknown')
        
    def load(self) -> bool:
        """
        Load custom model format.
        Override this method with your custom loading logic.
        
        Example implementations:
            - Load binary model weights
            - Initialize custom neural network
            - Load quantized models
            - Load ensemble models
        """
        self.logger.warning(f"CustomModelLoader.load() not implemented for format: {self.format}")
        self.logger.info("Override this method with your custom loading logic")
        
        # Example: Set default shapes
        self.input_shape = self.config.get('input_shape', (16, 96))
        self.output_shape = self.config.get('output_shape', (1,))
        
        return False
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run custom inference.
        Override this method with your custom prediction logic.
        
        Args:
            input_data: Input features
            
        Returns:
            np.ndarray: Predictions
        """
        self.logger.warning(f"CustomModelLoader.predict() not implemented for format: {self.format}")
        self.logger.info("Override this method with your custom inference logic")
        
        # Return dummy prediction
        return np.zeros(self.output_shape)
    
    def get_input_shape(self) -> tuple:
        return self.input_shape
    
    def get_output_shape(self) -> tuple:
        return self.output_shape


class ModelLoaderFactory:
    """
    Factory class for creating appropriate model loader instances.
    Implements the factory pattern for flexible model loading.
    """
    
    _loaders: Dict[str, type] = {
        'placeholder': PlaceholderModelLoader,
        'edge_impulse': EdgeImpulseModelLoader,
        'tflite': EdgeImpulseModelLoader,  # Alias
        'custom': CustomModelLoader,
    }
    
    @classmethod
    def register_loader(cls, name: str, loader_class: type):
        """
        Register a custom model loader.
        
        Args:
            name: Identifier for the loader
            loader_class: ModelLoader subclass
        """
        if not issubclass(loader_class, ModelLoader):
            raise ValueError(f"{loader_class} must be a subclass of ModelLoader")
        cls._loaders[name] = loader_class
    
    @classmethod
    def create_loader(cls, loader_type: str, model_path: str, 
                     config: Optional[Dict[str, Any]] = None) -> ModelLoader:
        """
        Create a model loader instance.
        
        Args:
            loader_type: Type of loader ('placeholder', 'edge_impulse', 'tflite', 'custom')
            model_path: Path to the model file
            config: Optional configuration dictionary
            
        Returns:
            ModelLoader: Instance of the appropriate loader class
            
        Raises:
            ValueError: If loader_type is not registered
        """
        if loader_type not in cls._loaders:
            raise ValueError(
                f"Unknown loader type: {loader_type}. "
                f"Available types: {list(cls._loaders.keys())}"
            )
        
        loader_class = cls._loaders[loader_type]
        return loader_class(model_path, config)
    
    @classmethod
    def get_available_loaders(cls) -> List[str]:
        """Get list of available loader types."""
        return list(cls._loaders.keys())
