import torch


@torch.no_grad()
def _binarize_predictions(logits, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    return preds


@torch.no_grad()
def dice_score(logits, targets, threshold=0.5, eps=1e-6):
    preds = _binarize_predictions(logits, threshold)

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    denom = preds.sum(dim=1) + targets.sum(dim=1)

    scores = (2 * intersection + eps) / (denom + eps)

    both_empty = (preds.sum(dim=1) == 0) & (targets.sum(dim=1) == 0)
    scores[both_empty] = 1.0
    return scores.mean().item()


@torch.no_grad()
def precision_score(logits, targets, threshold=0.5, eps=1e-6):
    preds = _binarize_predictions(logits, threshold)

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    tp = (preds * targets).sum(dim=1)
    fp = (preds * (1 - targets)).sum(dim=1)

    scores = (tp + eps) / (tp + fp + eps)

    both_empty = (preds.sum(dim=1) == 0) & (targets.sum(dim=1) == 0)
    scores[both_empty] = 1.0
    return scores.mean().item()


@torch.no_grad()
def recall_score(logits, targets, threshold=0.5, eps=1e-6):
    preds = _binarize_predictions(logits, threshold)

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    tp = (preds * targets).sum(dim=1)
    fn = ((1 - preds) * targets).sum(dim=1)

    scores = (tp + eps) / (tp + fn + eps)

    both_empty = (preds.sum(dim=1) == 0) & (targets.sum(dim=1) == 0)
    scores[both_empty] = 1.0
    return scores.mean().item()