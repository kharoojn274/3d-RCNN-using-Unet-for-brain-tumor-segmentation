import torch


def dice_per_class(pred, target, num_classes, eps=1e-6):
    pred = torch.argmax(pred, dim=1)
    scores = []
    for cls in range(1, num_classes):
        p = (pred == cls).float()
        t = (target == cls).float()
        inter = (p * t).sum()
        scores.append((2 * inter + eps) / (p.sum() + t.sum() + eps))
    return torch.stack(scores)
