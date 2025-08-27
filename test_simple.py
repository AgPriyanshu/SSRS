#!/usr/bin/env python3
"""Simple test script."""

print("Script started!")

try:
    from config import config
    print(f"Config loaded: {config.n_classes} classes")
    
    print("About to test model creation...")
    from model_wrapper import ModelWrapper
    
    print("Creating model...")
    model = ModelWrapper(num_classes=2)
    print("Model created successfully!")
    
except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()

print("Script finished!")
