import click
import torch

from rastervision.backend.torch_utils.metrics import (compute_conf_mat_metrics,
                                                      compute_conf_mat)


def train_epoch(model, device, data_loader, opt, loss_fn, step_scheduler=None):
    model.train()
    total_loss = 0.0
    num_samples = 0

    with click.progressbar(data_loader, label='Training') as bar:
        for batch_ind, (x, y) in enumerate(bar):
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            total_loss += loss.item()
            opt.step()
            if step_scheduler:
                step_scheduler.step()
            num_samples += x.shape[0]

    return total_loss / num_samples


def validate_epoch(model, device, data_loader, num_labels):
    model.eval()

    conf_mat = torch.zeros((num_labels, num_labels))
    with torch.no_grad():
        with click.progressbar(data_loader, label='Validating') as bar:
            for batch_ind, (x, y) in enumerate(bar):
                x = x.to(device)
                out = model(x)

                out = out.argmax(-1).view(-1).cpu()
                y = y.cpu()
                conf_mat += compute_conf_mat(out, y, num_labels)

    return compute_conf_mat_metrics(conf_mat)
