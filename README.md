# industrial-defect-detection
This repository contains the code and modules for our module framework built using the Real-IAD pre-trained model.

## Train the single-view baseline

From the repository root, run:

```bash
python3 src/train.py --epochs 10 --batch-size 8
```

The first epoch trains only the new classification and segmentation heads. Later
epochs also fine-tune ResNet50's `layer3` and `layer4` at one tenth of the main
learning rate. The checkpoint with the lowest validation loss is saved to
`checkpoints/best_single_view.pt`.

For a quick one-batch check before a full run:

```bash
python3 src/train.py --epochs 1 --batch-size 1 --max-batches 1
```
