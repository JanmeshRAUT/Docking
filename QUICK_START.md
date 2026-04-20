# Checkpoint System - Quick Start Guide

## 🚀 Quick Commands

### Start/Resume Training
```bash
python train_resnet50.py
```
- Automatically resumes from checkpoint if exists
- Otherwise starts from epoch 1
- Saves checkpoint after every epoch

### Check Current Progress
```bash
python check_progress.py
```
Shows:
- Current epoch vs. total epochs (progress bar)
- Current losses
- Early stopping status
- Improvement tips

### Check Checkpoint Details
```bash
python checkpoint_manager.py status
```
Shows:
- Last completed epoch
- Best validation loss
- Patience counter status
- Last training/validation losses

### Reset Training (Start Fresh)
```bash
python checkpoint_manager.py reset
```
Deletes checkpoint and optionally model weights.

---

## 📋 What Gets Saved

After each epoch, two files are automatically saved:

### 1. `resnet50_docking_best.pth` ⭐
- **Best model weights only**
- Saved when validation loss improves
- Use this for predictions

### 2. `training_checkpoint.pth` 💾
- **Complete training state**
- Saved after EVERY epoch
- Includes: epoch, losses, patience counter, optimizer state
- Use this to resume training exactly where you left off

---

## 🛟 If Training Gets Interrupted

### Scenario 1: You stop training (Ctrl+C)
```
Training stops...
Checkpoint is saved after current epoch finishes
```
Solution: Just run `python train_resnet50.py` again
→ Automatically resumes from last saved checkpoint

### Scenario 2: Power outage / System crash
```
Last complete epoch is in checkpoint
```
Solution: Run `python train_resnet50.py` again
→ Resumes from last checkpoint

### Scenario 3: You want to restart fresh
```
python checkpoint_manager.py reset
python train_resnet50.py
```
→ Starts from epoch 1

---

## 📊 Example: Resume Training

**First Run:**
```
python train_resnet50.py
Epoch [1/200]   | Train Loss: 0.003362 | Val Loss: 0.000297 ✓ Better weights saved | Checkpoint saved
...
Epoch [10/200]  | Train Loss: 0.000127 | Val Loss: 0.000067 ✓ Better weights saved | Checkpoint saved

[User stops training with Ctrl+C]
```

**Later (After a break):**
```
python train_resnet50.py

→ "CHECKPOINT FOUND! Resuming training..."
→ "Resuming from epoch 11/200"
→ "Best validation loss: 0.000067"
→ "Patience counter: 0/15"

[Training continues from epoch 11]
Epoch [11/200]  | Train Loss: 0.000125 | Val Loss: 0.000065 ✓ Better weights saved | Checkpoint saved
...
```

---

## ⚙️ Configuration

Edit `train_resnet50.py` to customize:

```python
AUTO_RESUME = True           # Auto-resume if checkpoint exists
EPOCHS = 200                 # Max epochs to train
EARLY_STOP_PATIENCE = 15     # Stop after 15 epochs without improvement
L2_REGULARIZATION = 1e-5     # Regularization strength (prevents overfitting)
LEARNING_RATE = 1e-4         # How fast the model learns
BATCH_SIZE = 32              # Samples per training step
```

---

## ✅ Best Practices

1. **Always check progress before stopping**
   ```bash
   python check_progress.py
   ```

2. **Monitor for overfitting**
   - If validation loss increases, model is overfitting
   - Early stopping will handle this (stops after 15 epochs)

3. **Save model periodically**
   - `resnet50_docking_best.pth` is your best model
   - Make backup copies for important runs

4. **Don't delete checkpoint files while training**
   - Safe to delete only after training completes

5. **Use early stopping wisely**
   - Patience = 15 means stop if no improvement for 15 epochs
   - Adjust if needed, but 15 is usually good

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Checkpoint not loading" | Run `python checkpoint_manager.py reset` |
| "CUDA out of memory" | Reduce `BATCH_SIZE` in train_resnet50.py |
| "Training stuck" | Check `python check_progress.py` and look at losses |
| "Can't resume" | Ensure `training_checkpoint.pth` exists in same folder |
| "Want fresh start" | Run `python checkpoint_manager.py reset` |

---

## 📁 Files Involved

```
e:\Docking\
├── train_resnet50.py              ← Main training script
├── checkpoint_manager.py           ← Manage checkpoints
├── check_progress.py               ← Check training progress
├── CHECKPOINT_GUIDE.md             ← Full documentation
│
├── resnet50_docking_best.pth       ← Best model (for inference)
├── training_checkpoint.pth         ← Complete checkpoint (for resuming)
│
└── Dataset/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

---

## 💡 Next Steps

1. **Start training:**
   ```bash
   python train_resnet50.py
   ```

2. **Monitor progress (while training):**
   ```bash
   python check_progress.py
   ```

3. **Make predictions after training:**
   ```bash
   python predict_and_visualize.py
   ```

Enjoy your checkpoint system! 🎉
