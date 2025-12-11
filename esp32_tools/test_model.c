/*
 * Test program for hey_assistant wake word model
 * Compile: gcc test_model.c -o test_model -lm
 * Run: ./test_model voice_samples/hey_assistant_000.wav
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "hey_assistant_model.h"

// Read WAV file (simple implementation, assumes standard 16-bit PCM WAV)
int16_t* read_wav_file(const char* filename, size_t* num_samples) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Cannot open file %s\n", filename);
        return NULL;
    }
    
    // Skip WAV header (44 bytes for standard format)
    fseek(file, 44, SEEK_SET);
    
    // Get file size
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 44, SEEK_SET);
    
    // Calculate number of samples
    *num_samples = (file_size - 44) / sizeof(int16_t);
    
    // Allocate buffer
    int16_t* samples = (int16_t*)malloc(*num_samples * sizeof(int16_t));
    if (!samples) {
        printf("Error: Memory allocation failed\n");
        fclose(file);
        return NULL;
    }
    
    // Read samples
    fread(samples, sizeof(int16_t), *num_samples, file);
    fclose(file);
    
    printf("Loaded %zu samples from %s\n", *num_samples, filename);
    return samples;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <wav_file>\n", argv[0]);
        printf("Example: %s voice_samples/hey_assistant_000.wav\n", argv[0]);
        return 1;
    }
    
    // Initialize model
    hey_assistant_model_t model;
    hey_assistant_init(&model);
    
    printf("Wake Word Detection Model: hey_assistant\n");
    printf("Threshold: %.2f\n", HEY_ASSISTANT_THRESHOLD);
    printf("Frame size: %d samples (80ms at 16kHz)\n\n", HEY_ASSISTANT_FRAME_SIZE);
    
    // Load audio file
    size_t num_samples;
    int16_t* audio_data = read_wav_file(argv[1], &num_samples);
    if (!audio_data) {
        return 1;
    }
    
    // Process audio in frames
    printf("Processing audio...\n\n");
    size_t frame_count = 0;
    size_t detection_count = 0;
    
    for (size_t i = 0; i + HEY_ASSISTANT_FRAME_SIZE <= num_samples; i += HEY_ASSISTANT_FRAME_SIZE) {
        float confidence;
        bool detected = hey_assistant_predict(&model, &audio_data[i], HEY_ASSISTANT_FRAME_SIZE, &confidence);
        
        frame_count++;
        
        if (detected) {
            detection_count++;
            printf("Frame %3zu: DETECTED! Confidence: %.2f, Energy: %.2f\n", 
                   frame_count, confidence, model.last_energy);
        } else {
            printf("Frame %3zu: -          Confidence: %.2f, Energy: %.2f\n", 
                   frame_count, confidence, model.last_energy);
        }
    }
    
    // Print statistics
    uint32_t total_frames, total_detections;
    float last_energy;
    hey_assistant_get_stats(&model, &total_frames, &total_detections, &last_energy);
    
    printf("\n--- Statistics ---\n");
    printf("Total frames: %u\n", total_frames);
    printf("Detections: %u (%.1f%%)\n", total_detections, 
           100.0f * total_detections / total_frames);
    printf("Last energy: %.2f\n", last_energy);
    
    // Cleanup
    free(audio_data);
    
    return 0;
}
