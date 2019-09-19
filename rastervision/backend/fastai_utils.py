import os
from os.path import join
import zipfile
from typing import Any
import warnings
import numpy as np

from fastai.callbacks import CSVLogger, Callback, SaveModelCallback, TrackerCallback
from fastai.metrics import add_metrics
from fastai.torch_core import dataclass, torch, Tensor, Optional, warn
from fastai.basic_train import Learner
from torch.utils.tensorboard import SummaryWriter

from fastai.vision import (LabelList,SegmentationItemList)

from rastervision.utils.files import (sync_to_dir)


class SyncCallback(Callback):
    """A callback to sync from_dir to to_uri at the end of epochs."""
    def __init__(self, from_dir, to_uri, sync_interval=1):
        self.from_dir = from_dir
        self.to_uri = to_uri
        self.sync_interval = sync_interval

    def on_epoch_end(self, **kwargs):
        if (kwargs['epoch'] + 1) % self.sync_interval == 0:
            sync_to_dir(self.from_dir, self.to_uri, delete=True)


class ExportCallback(TrackerCallback):
    """"Exports the model when monitored quantity is best.

    The exported model is the one used for inference.
    """
    def __init__(self, learn:Learner, model_path:str, monitor:str='valid_loss', mode:str='auto'):
        self.model_path = model_path
        super().__init__(learn, monitor=monitor, mode=mode)

    def on_epoch_end(self, epoch:int, **kwargs:Any)->None:
        current = self.get_monitor_value()

        if (epoch == 0 or
                (current is not None and self.operator(current, self.best))):
            print(f'Better model found at epoch {epoch} with {self.monitor} value: {current}.')
            self.best = current
            print(f'Exporting to {self.model_path}')
            self.learn.export(self.model_path)


class MySaveModelCallback(SaveModelCallback):
    """Saves the model after each epoch to potentially resume training.

    Modified from fastai version to delete the previous model that was saved
    to avoid wasting disk space.
    """
    def on_epoch_end(self, epoch:int, **kwargs:Any)->None:
        "Compare the value monitored to its best score and maybe save the model."
        if self.every=="epoch":
            self.learn.save(f'{self.name}_{epoch}')
            prev_model_path = self.learn.path/self.learn.model_dir/f'{self.name}_{epoch-1}.pth'
            if os.path.isfile(prev_model_path):
                os.remove(prev_model_path)
        else: #every="improvement"
            current = self.get_monitor_value()
            if current is not None and self.operator(current, self.best):
                print(f'Better model found at epoch {epoch} with {self.monitor} value: {current}.')
                self.best = current
                self.learn.save(f'{self.name}')


class MyCSVLogger(CSVLogger):
    """Logs metrics to a CSV file after each epoch.

    Modified from fastai version to:
    - flush after each epoch
    - append to log if already exists
    """
    def __init__(self, learn, filename='history'):
        super().__init__(learn, filename)

    def on_train_begin(self, **kwargs):
        if self.path.exists():
            self.file = self.path.open('a')
        else:
            super().on_train_begin(**kwargs)

    def on_epoch_end(self, epoch, smooth_loss, last_metrics, **kwargs):
        out = super().on_epoch_end(
            epoch, smooth_loss, last_metrics, **kwargs)
        self.file.flush()
        return out

# The following are a set of metric callbacks that have been modified from the
# original version in fastai to support semantic segmentation, which doesn't
# have the class dimension in position -1. It also adds an ignore_idx
# which is used to ignore pixels with class equal to ignore_idx. These
# would be good to contribute back upstream to fastai -- however we should
# wait for their upcoming refactor of the callback architecture.

@dataclass
class ConfusionMatrix(Callback):
    "Computes the confusion matrix."
    # The index of the dimension in the output and target arrays which ranges
    # over the different classes. This is -1 (the last index) for
    # classification, but is 1 for semantic segmentation.
    clas_idx:int=-1

    def on_train_begin(self, **kwargs):
        self.n_classes = 0

    def on_epoch_begin(self, **kwargs):
        self.cm = None

    def on_batch_end(self, last_output:Tensor, last_target:Tensor, **kwargs):
        preds = last_output.argmax(self.clas_idx).view(-1).cpu()
        targs = last_target.view(-1).cpu()
        if self.n_classes == 0:
            self.n_classes = last_output.shape[self.clas_idx]
            self.x = torch.arange(0, self.n_classes)
        cm = ((preds==self.x[:, None]) & (targs==self.x[:, None, None])).sum(dim=2, dtype=torch.float32)
        if self.cm is None: self.cm =  cm
        else:               self.cm += cm

    def on_epoch_end(self, **kwargs):
        self.metric = self.cm

@dataclass
class CMScores(ConfusionMatrix):
    "Base class for metrics which rely on the calculation of the precision and/or recall score."
    average:Optional[str]="binary"      # `binary`, `micro`, `macro`, `weighted` or None
    pos_label:int=1                     # 0 or 1
    eps:float=1e-9
    # If ground truth label is equal to the ignore_idx, it should be ignored
    # for the sake of evaluation.
    ignore_idx:int=None

    def _recall(self):
        rec = torch.diag(self.cm) / self.cm.sum(dim=1)
        rec[rec != rec] = 0  # removing potential "nan"s
        if self.average is None: return rec
        else:
            if self.average == "micro": weights = self._weights(avg="weighted")
            else: weights = self._weights(avg=self.average)
            return (rec * weights).sum()

    def _precision(self):
        prec = torch.diag(self.cm) / self.cm.sum(dim=0)
        prec[prec != prec] = 0  # removing potential "nan"s
        if self.average is None: return prec
        else:
            weights = self._weights(avg=self.average)
            return (prec * weights).sum()

    def _weights(self, avg:str):
        if self.n_classes != 2 and avg == "binary":
            avg = self.average = "macro"
            warn("average=`binary` was selected for a non binary case. Value for average has now been set to `macro` instead.")
        if avg == "binary":
            if self.pos_label not in (0, 1):
                self.pos_label = 1
                warn("Invalid value for pos_label. It has now been set to 1.")
            if self.pos_label == 1: return Tensor([0,1])
            else: return Tensor([1,0])
        else:
            if avg == "micro": weights = self.cm.sum(dim=0) / self.cm.sum()
            if avg == "macro": weights = torch.ones((self.n_classes,)) / self.n_classes
            if avg == "weighted": weights = self.cm.sum(dim=1) / self.cm.sum()
            if self.ignore_idx is not None and avg in ["macro", "weighted"]:
                weights[self.ignore_idx] = 0
                weights /= weights.sum()
            return weights

class Recall(CMScores):
    "Compute the Recall."
    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics, self._recall())

class Precision(CMScores):
    "Compute the Precision."
    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics, self._precision())

@dataclass
class FBeta(CMScores):
    "Compute the F`beta` score."
    beta:float=2

    def on_train_begin(self, **kwargs):
        self.n_classes = 0
        self.beta2 = self.beta ** 2
        self.avg = self.average
        if self.average != "micro": self.average = None

    def on_epoch_end(self, last_metrics, **kwargs):
        prec = self._precision()
        rec = self._recall()
        metric = (1 + self.beta2) * prec * rec / (prec * self.beta2 + rec + self.eps)
        metric[metric != metric] = 0  # removing potential "nan"s
        if self.avg: metric = (self._weights(avg=self.avg) * metric).sum()
        return add_metrics(last_metrics, metric)

    def on_train_end(self, **kwargs): self.average = self.avg


def zipdir(dir, zip_path):
    """Create a zip file from a directory.

    The zip file contains the contents of dir, but not dir itself.

    Args:
        dir: (str) the directory with the content to place in zip file
        zip_path: (str) path to the zip file
    """
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as ziph:
        for root, dirs, files in os.walk(dir):
            for file in files:
                ziph.write(join(root, file),
                           join('/'.join(dirs),
                                os.path.basename(file)))

def get_oversampling_weights(
    dataset,
    rare_target_prop,
    rare_class_ids=None,
    rare_class_names=None):
    """Return weight vector for oversampling chips with rare classes.

    Args:
        dataset: PyTorch DataSet with semantic segmentation or chip classification data
        rare_class_ids: list of rare class ids, in case of semantic segmentation
        rare_class_names: list of rare class names, in case of chip classification
        rare_target_prop: desired probability of sampling a chip covering the
            rare classes
    """

    # Check that either the id or name is given, not both or neither
    if rare_class_ids != None and rare_class_names != None:
        log.error("You are using oversampling and have \
            specified both rare_class_ids as well as rare_class_names \
            You can only specify one.")
    elif rare_class_ids == None and rare_class_names == None:
        log.error("You are using oversampling but have not specified \
            rare_class_names or rare_class_ids. You should specify rare_class_names \
            if you are using chip classification, rare_class_ids if doing semantic\
            segmentation")

    def filter_chip_inds_byclassid():
        chip_inds = []
        for i, (x, y) in enumerate(dataset):
            match = False
            for class_id in rare_class_ids:
                if torch.any(torch.from_numpy(np.array(y.data)) == class_id):
                    match = True
                    break
            if match:
                chip_inds.append(i)
        return chip_inds

    def filter_chip_inds_byname():
        '''
        For the chip segmentation the labels have two attributes:
        "data", which is the index of the folder the image is in
        and "obj", which is a string of the folder name the image is in.
        Using the indices is not safe and can be confusing for the user
        as the task.with_classes() method allows the user to specify
        an id per class, which does not have to match the index of the folder.
        Therefore it is safer to use the name of the folder instead of the index.
        '''
        chip_inds = []
        for i, (x, y) in enumerate(dataset):
            match = False
            for class_name in rare_class_names:
                if y.obj == class_name:
                    math = True
                    break
            if match:
                chip_inds.append(i)
        return chip_inds

    def get_sample_weights(num_samples, rare_chip_inds, rare_target_prob):
        rare_weight = rare_target_prob / len(rare_chip_inds)
        common_weight = (1 - rare_target_prob) / (
            num_samples - len(rare_chip_inds))
        weights = torch.full((num_samples, ), common_weight)
        weights[rare_chip_inds] = rare_weight
        return weights

    if rare_class_ids: # It is a segmenation task, filter by class id
        chip_inds = filter_chip_inds_byclassid()
    elif rare_class_names: # it is a chip classification task, filter by class name
        chip_inds = filter_chip_inds_byname()

    print('prop of rare chips before oversampling: ',
          len(chip_inds) / len(dataset))
    weights = get_sample_weights(len(dataset), chip_inds, rare_target_prop)
    sys.exit()
    return weights



# This code was adapted from
# https://github.com/Pendar2/fastai-tensorboard-callback/blob/master/fastai_tensorboard_callback/tensorboard_cb.py
@dataclass
class TensorboardLogger(Callback):
    learn:Learner
    run_name:str
    histogram_freq:int=100
    path:str=None

    def __post_init__(self):
        self.path = self.path or os.path.join(self.learn.path, "logs")
        self.log_dir = os.path.join(self.path, self.run_name)

    def on_train_begin(self, **kwargs):
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def on_epoch_end(self, **kwargs):
        iteration = kwargs["iteration"]
        metrics = kwargs["last_metrics"]
        metrics_names = ["valid_loss"] + [o.__class__.__name__ for o in self.learn.metrics]

        for val, name in zip(metrics, metrics_names):
            self.writer.add_scalar(name, val, iteration)

    def on_batch_end(self, **kwargs):
        iteration = kwargs["iteration"]
        loss = kwargs["last_loss"]

        self.writer.add_scalar("learning_rate", self.learn.opt.lr, iteration)
        self.writer.add_scalar("momentum", self.learn.opt.mom, iteration)
        self.writer.add_scalar("loss", loss, iteration)

        if iteration%self.histogram_freq == 0:
            for name, param in self.learn.model.named_parameters():
                self.writer.add_histogram(name, param, iteration)

    def on_train_end(self, **kwargs):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dummy_input = next(iter(self.learn.data.train_dl))[0]
                self.writer.add_graph(self.learn.model, tuple(dummy_input))
        except Exception as e:
            print("Unable to create graph.")
            print(e)
        self.writer.close()