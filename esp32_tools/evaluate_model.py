#!/usr/bin/env python3
"""
Evaluate wake word detection model effectiveness.
Measures accuracy, false positive rate, and detection latency.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
import subprocess
import json

def evaluate_on_positive_samples(test_program, voice_dir):
    """
    Test on positive samples (actual wake word recordings).
    
    Returns:
        dict: Detection statistics
    """
    print("=" * 60)
    print("POSITIVE SAMPLES EVALUATION (Wake Word Present)")
    print("=" * 60)
    
    wav_files = sorted(list(Path(voice_dir).glob('*.wav')))
    
    results = {
        'total_files': len(wav_files),
        'detected_files': 0,
        'total_frames': 0,
        'detected_frames': 0,
        'detection_rates': [],
        'energies': []
    }
    
    for wav_file in wav_files:
        # Run test program
        try:
            output = subprocess.check_output(
                [test_program, str(wav_file)],
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Parse output
            detected_frames = 0
            total_frames = 0
            energies = []
            
            for line in output.split('\n'):
                if line.startswith('Frame'):
                    total_frames += 1
                    if 'DETECTED!' in line:
                        detected_frames += 1
                    # Extract energy
                    if 'Energy:' in line:
                        energy = float(line.split('Energy:')[1].strip())
                        energies.append(energy)
            
            if detected_frames > 0:
                results['detected_files'] += 1
            
            results['total_frames'] += total_frames
            results['detected_frames'] += detected_frames
            
            detection_rate = (detected_frames / total_frames * 100) if total_frames > 0 else 0
            results['detection_rates'].append(detection_rate)
            results['energies'].extend(energies)
            
            status = "✓ DETECTED" if detected_frames > 0 else "✗ MISSED"
            print(f"{wav_file.name:30} {status:12} ({detected_frames}/{total_frames} frames, {detection_rate:.1f}%)")
            
        except subprocess.CalledProcessError as e:
            print(f"{wav_file.name:30} ERROR: {e}")
    
    # Calculate metrics
    file_detection_rate = (results['detected_files'] / results['total_files'] * 100) if results['total_files'] > 0 else 0
    frame_detection_rate = (results['detected_frames'] / results['total_frames'] * 100) if results['total_frames'] > 0 else 0
    
    print("\n" + "-" * 60)
    print(f"Files with detections:  {results['detected_files']}/{results['total_files']} ({file_detection_rate:.1f}%)")
    print(f"Total frame detections: {results['detected_frames']}/{results['total_frames']} ({frame_detection_rate:.1f}%)")
    print(f"Avg detection rate:     {np.mean(results['detection_rates']):.1f}% per file")
    print(f"Min/Max detection:      {np.min(results['detection_rates']):.1f}% / {np.max(results['detection_rates']):.1f}%")
    
    if results['energies']:
        print(f"\nEnergy statistics:")
        print(f"  Mean:   {np.mean(results['energies']):.2f}")
        print(f"  Median: {np.median(results['energies']):.2f}")
        print(f"  Std:    {np.std(results['energies']):.2f}")
        print(f"  Range:  {np.min(results['energies']):.2f} - {np.max(results['energies']):.2f}")
    
    return results

def evaluate_on_negative_samples(test_program, negative_dir):
    """
    Test on negative samples (background noise, other sounds).
    
    Returns:
        dict: False positive statistics
    """
    print("\n" + "=" * 60)
    print("NEGATIVE SAMPLES EVALUATION (No Wake Word)")
    print("=" * 60)
    
    if not Path(negative_dir).exists():
        print(f"No negative samples directory found: {negative_dir}")
        print("Skipping false positive evaluation.")
        print("\nTo test false positives, create a directory with:")
        print("  - Background noise recordings")
        print("  - Other speech recordings")
        print("  - Environmental sounds")
        return None
    
    wav_files = sorted(list(Path(negative_dir).glob('*.wav')))
    
    if not wav_files:
        print(f"No WAV files found in {negative_dir}")
        return None
    
    results = {
        'total_files': len(wav_files),
        'false_positive_files': 0,
        'total_frames': 0,
        'false_positive_frames': 0,
        'false_positive_rates': []
    }
    
    for wav_file in wav_files:
        try:
            output = subprocess.check_output(
                [test_program, str(wav_file)],
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Parse output
            detected_frames = 0
            total_frames = 0
            
            for line in output.split('\n'):
                if line.startswith('Frame'):
                    total_frames += 1
                    if 'DETECTED!' in line:
                        detected_frames += 1
            
            if detected_frames > 0:
                results['false_positive_files'] += 1
            
            results['total_frames'] += total_frames
            results['false_positive_frames'] += detected_frames
            
            fp_rate = (detected_frames / total_frames * 100) if total_frames > 0 else 0
            results['false_positive_rates'].append(fp_rate)
            
            status = "✗ FALSE POS" if detected_frames > 0 else "✓ CORRECT"
            print(f"{wav_file.name:30} {status:12} ({detected_frames}/{total_frames} frames, {fp_rate:.1f}%)")
            
        except subprocess.CalledProcessError as e:
            print(f"{wav_file.name:30} ERROR: {e}")
    
    # Calculate metrics
    file_fp_rate = (results['false_positive_files'] / results['total_files'] * 100) if results['total_files'] > 0 else 0
    frame_fp_rate = (results['false_positive_frames'] / results['total_frames'] * 100) if results['total_frames'] > 0 else 0
    
    print("\n" + "-" * 60)
    print(f"Files with false positives: {results['false_positive_files']}/{results['total_files']} ({file_fp_rate:.1f}%)")
    print(f"Total false positive frames: {results['false_positive_frames']}/{results['total_frames']} ({frame_fp_rate:.1f}%)")
    print(f"Avg false positive rate:    {np.mean(results['false_positive_rates']):.1f}% per file")
    
    return results

def calculate_metrics(positive_results, negative_results):
    """Calculate overall performance metrics."""
    print("\n" + "=" * 60)
    print("OVERALL PERFORMANCE METRICS")
    print("=" * 60)
    
    # True Positive Rate (Sensitivity/Recall)
    tpr = (positive_results['detected_files'] / positive_results['total_files'] * 100)
    print(f"True Positive Rate (TPR):     {tpr:.1f}%")
    print(f"  = Files correctly detected / Total wake word files")
    
    if negative_results:
        # False Positive Rate
        fpr = (negative_results['false_positive_files'] / negative_results['total_files'] * 100)
        print(f"\nFalse Positive Rate (FPR):    {fpr:.1f}%")
        print(f"  = Files incorrectly detected / Total non-wake-word files")
        
        # Specificity
        specificity = 100 - fpr
        print(f"\nSpecificity:                  {specificity:.1f}%")
        print(f"  = Correctly rejected non-wake-word files")
        
        # Precision (if we have both metrics)
        tp = positive_results['detected_files']
        fp = negative_results['false_positive_files']
        if tp + fp > 0:
            precision = (tp / (tp + fp) * 100)
            print(f"\nPrecision:                    {precision:.1f}%")
            print(f"  = True positives / (True positives + False positives)")
        
        # F1 Score
        if tpr > 0 and precision > 0:
            f1 = 2 * (precision * tpr) / (precision + tpr)
            print(f"\nF1 Score:                     {f1:.1f}%")
            print(f"  = Harmonic mean of precision and recall")
    
    # Performance assessment
    print("\n" + "=" * 60)
    print("EFFECTIVENESS ASSESSMENT")
    print("=" * 60)
    
    if tpr >= 95:
        print("✓ EXCELLENT: Detects almost all wake words (≥95%)")
    elif tpr >= 85:
        print("✓ GOOD: Detects most wake words (≥85%)")
    elif tpr >= 70:
        print("⚠ FAIR: Misses some wake words (70-85%)")
    else:
        print("✗ POOR: Misses too many wake words (<70%)")
    
    if negative_results:
        if fpr <= 5:
            print("✓ EXCELLENT: Very few false alarms (≤5%)")
        elif fpr <= 15:
            print("✓ GOOD: Acceptable false alarm rate (≤15%)")
        elif fpr <= 30:
            print("⚠ FAIR: Moderate false alarms (15-30%)")
        else:
            print("✗ POOR: Too many false alarms (>30%)")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    if tpr < 85:
        print("• Lower threshold to increase sensitivity")
        print("  python3 generate_model_c_code.py --threshold-percentile 50")
    
    if negative_results and fpr > 15:
        print("• Increase threshold to reduce false positives")
        print("  python3 generate_model_c_code.py --threshold-percentile 90")
    
    if tpr >= 85 and (not negative_results or fpr <= 15):
        print("✓ Model is well-balanced and effective for deployment")
    
    print("\n• Collect more training samples to improve robustness")
    print("• Test with background noise and other speakers")
    print("• Consider using ML model for better accuracy")

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate wake word detection model effectiveness'
    )
    parser.add_argument(
        '--test-program',
        type=str,
        default='./test_model',
        help='Path to compiled test program (default: ./test_model)'
    )
    parser.add_argument(
        '--positive-dir',
        type=str,
        default='voice_samples',
        help='Directory with wake word samples (default: voice_samples)'
    )
    parser.add_argument(
        '--negative-dir',
        type=str,
        default='negative_samples',
        help='Directory with non-wake-word samples (default: negative_samples)'
    )
    
    args = parser.parse_args()
    
    # Check if test program exists
    if not Path(args.test_program).exists():
        print(f"Error: Test program not found: {args.test_program}")
        print("Compile it first: gcc test_model.c -o test_model -lm")
        sys.exit(1)
    
    # Evaluate on positive samples
    positive_results = evaluate_on_positive_samples(args.test_program, args.positive_dir)
    
    # Evaluate on negative samples
    negative_results = evaluate_on_negative_samples(args.test_program, args.negative_dir)
    
    # Calculate overall metrics
    calculate_metrics(positive_results, negative_results)
    
    # Save results
    results = {
        'positive': positive_results,
        'negative': negative_results
    }
    
    with open('evaluation_results.json', 'w') as f:
        # Convert numpy types to native Python types
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            return obj
        
        json.dump(convert(results), f, indent=2)
    
    print(f"\nResults saved to: evaluation_results.json")

if __name__ == '__main__':
    main()
