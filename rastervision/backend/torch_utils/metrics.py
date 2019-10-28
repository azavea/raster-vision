import torch


def compute_conf_mat(out, y, num_labels):
    labels = torch.arange(0, num_labels)
    return ((out == labels[:, None]) & (y == labels[:, None, None])).sum(
        dim=2, dtype=torch.float32)


def compute_conf_mat_metrics(conf_mat, eps=1e-6):
    # eps is to avoid dividing by zero.
    eps = torch.tensor(eps)
    gt_count = conf_mat.sum(dim=1)
    pred_count = conf_mat.sum(dim=0)
    total = conf_mat.sum()
    true_pos = torch.diag(conf_mat)
    precision = true_pos / torch.max(pred_count, eps)
    recall = true_pos / torch.max(gt_count, eps)

    weights = gt_count / total
    weighted_precision = (weights * precision).sum()
    weighted_recall = (weights * recall).sum()
    weighted_f1 = ((2 * weighted_precision * weighted_recall) / torch.max(
        weighted_precision + weighted_recall, eps))
    metrics = {
        'precision': weighted_precision.item(),
        'recall': weighted_recall.item(),
        'f1': weighted_f1.item()
    }
    return metrics
