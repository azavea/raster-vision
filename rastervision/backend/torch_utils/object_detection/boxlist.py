from collections import defaultdict

import torch

from torchvision.ops.boxes import batched_nms


def to_box_pixel(boxes, img_height, img_width):
    # convert from (ymin, xmin, ymax, xmax) in range [-1,1] to
    # range [0, h) or [0, w)
    boxes = ((boxes + 1.0) / 2.0) * torch.tensor(
        [[img_height, img_width, img_height, img_width]]).to(
            device=boxes.device, dtype=torch.float)
    return boxes


class BoxList():
    def __init__(self, boxes, **extras):
        """Constructor.

        Args:
            boxes: tensor<n, 4> with order ymin, xmin, ymax, xmax in pixels coords
            extras: dict with values that are tensors with first dimension corresponding
                to boxes first dimension
        """
        self.boxes = boxes
        self.extras = extras

    def get_field(self, name):
        if name == 'boxes':
            return self.boxes
        else:
            return self.extras.get(name)

    def _map_extras(self, func):
        new_extras = {}
        for k, v in self.extras.items():
            new_extras[k] = func(v)
        return new_extras

    def copy(self):
        return BoxList(self.boxes.copy(),
                       **self._map_extras(lambda x: x.copy()))

    def cpu(self):
        return BoxList(self.boxes.cpu(), **self._map_extras(lambda x: x.cpu()))

    def cuda(self):
        return BoxList(self.boxes.cuda(),
                       **self._map_extras(lambda x: x.cuda()))

    def to(self, device):
        return self.cpu() if device == 'cpu' else self.cuda()

    def xyxy(self):
        boxes = self.boxes[:, [1, 0, 3, 2]]
        return BoxList(boxes, **self.extras)

    def yxyx(self):
        boxes = self.boxes[:, [1, 0, 3, 2]]
        return BoxList(boxes, **self.extras)

    def __len__(self):
        return self.boxes.shape[0]

    @staticmethod
    def cat(box_lists):
        boxes = []
        extras = defaultdict(list)
        for bl in box_lists:
            boxes.append(bl.boxes)
            for k, v in bl.extras.items():
                extras[k].append(v)
        boxes = torch.cat(boxes)
        for k, v in extras.items():
            extras[k] = torch.cat(v)
        return BoxList(boxes, **extras)

    def equal(self, other):
        if len(other) != len(self):
            return False

        # Ignore order of boxes.
        extras = [(v.float().unsqueeze(1) if v.ndim == 1 else v.float())
                  for v in self.extras.values()]
        cat_arr = torch.cat([self.boxes] + extras, 1)
        self_tups = set([tuple([x.item() for x in row]) for row in cat_arr])

        extras = [(v.float().unsqueeze(1) if v.ndim == 1 else v.float())
                  for v in other.extras.values()]
        cat_arr = torch.cat([other.boxes] + extras, 1)
        other_tups = set([tuple([x.item() for x in row]) for row in cat_arr])
        return self_tups == other_tups

    def ind_filter(self, inds):
        new_extras = {}
        for k, v in self.extras.items():
            new_extras[k] = v[inds, ...]
        return BoxList(self.boxes[inds, :], **new_extras)

    def score_filter(self, score_thresh=0.25):
        scores = self.extras.get('scores')
        if scores is not None:
            return self.ind_filter(scores > score_thresh)
        else:
            raise ValueError('must have scores as key in extras')

    def clamp(self, img_height, img_width):
        boxes = torch.stack(
            [
                torch.clamp(self.boxes[:, 0], 0, img_height),
                torch.clamp(self.boxes[:, 1], 0, img_width),
                torch.clamp(self.boxes[:, 2], 0, img_height),
                torch.clamp(self.boxes[:, 3], 0, img_width)
            ],
            dim=1)
        return BoxList(boxes, **self.extras)

    def nms(self, iou_thresh=0.5):
        if len(self) == 0:
            return self

        good_inds = batched_nms(self.boxes, self.get_field('scores'),
                                self.get_field('labels'), iou_thresh)
        return self.ind_filter(good_inds)

    def scale(self, yscale, xscale):
        boxes = self.boxes * torch.tensor(
            [[yscale, xscale, yscale, xscale]], device=self.boxes.device)
        return BoxList(boxes, **self.extras)

    def pin_memory(self):
        self.boxes = self.boxes.pin_memory()
        for k, v in self.extras.items():
            self.extras[k] = v.pin_memory()
        return self
