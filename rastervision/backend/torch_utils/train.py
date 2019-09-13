from collections import defaultdict
import warnings

import click
import torch

from rastervision.backend.torch_utils.metrics import compute_coco_eval

warnings.filterwarnings('ignore')


def train_epoch(model,
                device,
                dl,
                opt,
                step_scheduler=None,
                epoch_scheduler=None):
    model.train()
    train_loss = defaultdict(lambda: 0.0)
    num_samples = 0

    with click.progressbar(dl, label='Training') as bar:
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


def validate_epoch(model, device, dl, num_labels):
    model.eval()

    ys = []
    outs = []
    with torch.no_grad():
        with click.progressbar(dl, label='Validating') as bar:
            for batch_ind, (x, y) in enumerate(bar):
                x = x.to(device)
                out = model(x)

                ys.extend([_y.cpu() for _y in y])
                outs.extend([_out.cpu() for _out in out])

    coco_metrics = compute_coco_eval(outs, ys, num_labels)
    metrics = {'map': coco_metrics[0], 'map50': coco_metrics[1]}
    return metrics
