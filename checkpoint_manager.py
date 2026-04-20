"""
Checkpoint Management Utility
Allows checking, resuming, or resetting training checkpoints
"""

import torch
import os

CHECKPOINT_PATH = "training_checkpoint.pth"
MODEL_SAVE_PATH = "resnet50_docking_best.pth"

def show_checkpoint_status():
    """Display current checkpoint and model status."""
    print("\n" + "="*70)
    print("TRAINING STATUS CHECK")
    print("="*70)
    
    checkpoint_exists = os.path.exists(CHECKPOINT_PATH)
    model_exists = os.path.exists(MODEL_SAVE_PATH)
    
    print(f"Model file exists: {model_exists} ({MODEL_SAVE_PATH})")
    print(f"Checkpoint file exists: {checkpoint_exists} ({CHECKPOINT_PATH})")
    
    if checkpoint_exists:
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
            print("\n" + "-"*70)
            print("CHECKPOINT DETAILS:")
            print("-"*70)
            print(f"  Last Epoch Completed: {checkpoint['epoch']}")
            print(f"  Best Validation Loss: {checkpoint['best_loss']:.6f}")
            print(f"  Patience Counter: {checkpoint['patience_counter']}/15")
            print(f"  Last Training Loss: {checkpoint['train_loss']:.6f}")
            print(f"  Last Validation Loss: {checkpoint['val_loss']:.6f}")
            print("-"*70)
            print("\n✓ You can resume training by running: python train_resnet50.py")
        except Exception as e:
            print(f"\n⚠ Error reading checkpoint: {e}")
    else:
        print("\n⚠ No checkpoint found. Training will start from epoch 1.")
    
    print("="*70 + "\n")

def reset_checkpoint():
    """Delete checkpoint and start fresh."""
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print(f"✓ Checkpoint deleted: {CHECKPOINT_PATH}")
        print("  Next training will start from epoch 1.")
    else:
        print("No checkpoint to delete.")
    
    if os.path.exists(MODEL_SAVE_PATH):
        response = input(f"\nDelete model weights too? ({MODEL_SAVE_PATH}) (y/n): ")
        if response.lower() == 'y':
            os.remove(MODEL_SAVE_PATH)
            print(f"✓ Model weights deleted: {MODEL_SAVE_PATH}")
        else:
            print("Model weights retained.")
    print()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "reset":
            print("\n⚠ WARNING: This will delete the checkpoint and restart training!")
            response = input("Are you sure? (yes/no): ")
            if response.lower() == "yes":
                reset_checkpoint()
            else:
                print("Cancelled.")
        elif sys.argv[1] == "status":
            show_checkpoint_status()
        else:
            print(f"Unknown command: {sys.argv[1]}")
            print("Usage:")
            print("  python checkpoint_manager.py status  - Show checkpoint status")
            print("  python checkpoint_manager.py reset   - Delete checkpoint and start fresh")
    else:
        show_checkpoint_status()
