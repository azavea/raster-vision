from collections import defaultdict
import warnings

import click
import torch
import numpy as np

from rastervision.backend.torch_utils.object_detection.metrics import (
    compute_coco_eval, compute_class_f1)

warnings.filterwarnings('ignore')


def train_epoch(model,
                device,
                data_loader,
                opt,
                step_scheduler=None,
                epoch_scheduler=None):
    model.train()
    train_loss = defaultdict(lambda: 0.0)
    num_samples = 0

    with click.progressbar(data_loader, label='Training') as bar:
        for batch_ind, (x, y) in enumerate(bar):
            x = x.to(device)
            y = [_y.to(device) for _y in y]

            opt.zero_grad()
            loss_dict = model(x, y)
            loss_dict['total_loss'].backward()
            opt.step()
            if step_scheduler:
                step_scheduler.step()

            for k, v in loss_dict.items():
                train_loss[k] += v.item()
            num_samples += x.shape[0]

    for k, v in train_loss.items():
        train_loss[k] = v / num_samples

    return dict(train_loss)


def validate_epoch(model, device, data_loader, num_labels):
    model.eval()

    ys = []
    outs = []
    with torch.no_grad():
        with click.progressbar(data_loader, label='Validating') as bar:
            for batch_ind, (x, y) in enumerate(bar):
                x = x.to(device)
                out = model(x)

                ys.extend([_y.cpu() for _y in y])
                outs.extend([_out.cpu() for _out in out])

    coco_eval = compute_coco_eval(outs, ys, num_labels)

    metrics = {
        'map': 0.0,
        'map50': 0.0,
        'mean_f1': 0.0,
        'mean_score_thresh': 0.5
    }
    if coco_eval is not None:
        coco_metrics = coco_eval.stats
        best_f1s, best_scores = compute_class_f1(coco_eval)
        mean_f1 = np.mean(best_f1s[1:])
        mean_score_thresh = np.mean(best_scores[1:])
        metrics = {
            'map': coco_metrics[0],
            'map50': coco_metrics[1],
            'mean_f1': mean_f1,
            'mean_score_thresh': mean_score_thresh
        }
    return metrics
