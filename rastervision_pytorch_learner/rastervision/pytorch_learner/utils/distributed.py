from typing import TYPE_CHECKING, Any, Optional
import os
from contextlib import AbstractContextManager
import gc
import logging

import torch
import torch.distributed as dist

from rastervision.pipeline import rv_config_ as rv_config

if TYPE_CHECKING:
    from rastervision.pytorch_learner import Learner

log = logging.getLogger(__name__)

DDP_BACKEND = rv_config.get_namespace_option('rastervision', 'DDP_BACKEND',
                                             'nccl')


class DDPContextManager(AbstractContextManager):  # pragma: no cover
    """Context manager for initializing and destroying DDP process groups.

    Note that this context manager does not start processes itself, but
    merely calls :func:`torch.distributed.init_process_group` and
    :func:`torch.distributed.destroy_process_group` and sets DDP-related fields
    in the :class:`Learner` to appropriate values.

    If a process group is already initialized, this context manager does
    nothing on either entry or exit.
    """

    def __init__(self,
                 learner: 'Learner',
                 rank: Optional[int] = None,
                 world_size: Optional[int] = None) -> None:
        """Constructor.

        Args:
            learner: The :class:`Learner` on which to set DDP-related fields.
            rank: The process rank. If ``None``, will be set to
                ``Learner.ddp_rank``. Defaults to ``None``.
            world_size: The world size. If ``None``, will be set to
                ``Learner.ddp_world_size``. Defaults to ``None``.

        Raises:
            ValueError: If ``rank`` or ``world_size`` not provided and aren't
                set on the :class:`Learner`.
        """
        self.learner = learner
        self.rank = learner.ddp_rank if rank is None else rank
        self.world_size = (learner.ddp_world_size
                           if world_size is None else world_size)
        if self.rank is None or self.world_size is None:
            raise ValueError('Could not determine rank and world_size.')
        self.noop = dist.is_initialized()

    def __enter__(self) -> Any:
        if self.noop:
            return

        learner = self.learner
        rank = self.rank
        world_size = self.world_size

        os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')

        log.debug('Calling init_process_group()')
        dist.init_process_group(DDP_BACKEND, rank=rank, world_size=world_size)

        if learner.ddp_rank is None:
            learner.ddp_rank = rank
        if learner.ddp_local_rank is None:
            # Implies process was spawned by learner.train(), and therefore,
            # this is necessarily a single-node multi-GPU scenario.
            # So global rank == local rank.
            learner.ddp_local_rank = rank
        if learner.ddp_world_size is None:
            learner.ddp_world_size = world_size

        log.info('DDP rank: %d, DDP local rank: %d', learner.ddp_rank,
                 learner.ddp_local_rank)

        learner.is_ddp_process = True
        learner.is_ddp_master = learner.ddp_rank == 0
        learner.is_ddp_local_master = learner.ddp_local_rank == 0

        learner.device = torch.device(learner.device.type,
                                      learner.ddp_local_rank)
        torch.cuda.set_device(learner.device)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.noop:
            return
        dist.barrier()
        dist.destroy_process_group()
        gc.collect()
