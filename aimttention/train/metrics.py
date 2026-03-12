from ignite.engine import Engine
import torch
from torch import Tensor
import ignite.distributed as idist
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import reinit__is_reduced
from typing import List, Dict
from collections import defaultdict
import logging


class RegMultiMetric(Metric):
    def __init__(self, cfg : List[Dict], loss_fn=None):
        super().__init__()
        self.cfg = cfg
        self.loss_fn = loss_fn

    def attach_loss(self, loss_fn):
        self.loss_fn = loss_fn

    def attach(self, engine: Engine, name: str = ''):
        # Override to flatten dict metrics into engine.state.metrics
        # so wandb/ignite see individual metric names instead of a nested dict.
        from ignite.engine import Events
        super().attach(engine, name)
        def _flatten_metrics(engine):
            val = engine.state.metrics.pop(name, {})
            if isinstance(val, dict):
                engine.state.metrics.update(val)
        engine.add_event_handler(Events.EPOCH_COMPLETED, _flatten_metrics)

    @reinit__is_reduced
    def reset(self):
        super().reset()
        self.data = defaultdict(lambda: defaultdict(float))
        self.atoms = 0.0
        self.samples = 0.0
        self.loss = defaultdict(float)

    def _update_one(self, key: str, pred: Tensor, true: Tensor) -> None:
        e = true - pred
        if pred.ndim > true.ndim:
            e = e.view(pred.shape[0], -1)
        else:
            e = e.view(-1)
        d = self.data[key]
        d['sum_abs_err'] += e.abs().sum(-1).cpu().float()
        d['sum_sq_err'] += e.pow(2).sum(-1).cpu().float()
        d['sum_true'] += true.sum().cpu().float()
        d['sum_sq_true'] += true.pow(2).sum().cpu().float()

    @reinit__is_reduced
    def update(self, output) -> None:
        y_pred, y_true = output
        if y_pred is None:
            return
        for k in y_pred:
            if k not in y_true:
                continue
            with torch.no_grad():
                self._update_one(k, y_pred[k].detach(), y_true[k].detach())
            b = y_true[k].shape[0]
        self.samples += b

        _n = y_pred['_natom']
        if _n.numel() > 1:
            self.atoms += _n.sum().item()
        else:
            self.atoms += y_pred['numbers'].shape[0] * y_pred['numbers'].shape[1]
        if self.loss_fn is not None:
            with torch.no_grad():
                loss_d = self.loss_fn(y_pred, y_true)
                for k, loss in loss_d.items():
                    if isinstance(loss, Tensor):
                        if loss.numel() > 1:
                            loss = loss.mean()
                        loss = loss.item()
                    self.loss[k] += loss * b

    def compute(self):
        if self.samples == 0:
            raise NotComputableError
        if idist.get_world_size() > 1:
            self.atoms = idist.all_reduce(self.atoms)
            self.samples = idist.all_reduce(self.samples)
            for k, loss in self.loss.items():
                self.loss[k] = idist.all_reduce(loss)
            for k1, v1 in self.data.items():
                for k2, v2 in v1.items():
                    self.data[k1][k2] = idist.all_reduce(v2)
        self._is_reduced = True

        ret = dict()
        for k in self.data:
            if k not in self.cfg:
                continue
            cfg = self.cfg[k]
            _n = self.atoms if cfg.get('peratom', False) else self.samples
            _n *= cfg.get('mult', 1.0)
            name = k
            abbr = cfg['abbr']
            v = self.data[name]
            m = dict()
            m['mae'] = v['sum_abs_err'] / _n
            m['rmse'] = (v['sum_sq_err'] / _n).sqrt()
            m['r2'] = 1.0 - v['sum_sq_err'] / (v['sum_sq_true'] - (v['sum_true'].pow(2)) / _n)
            for k, v in m.items():
                if k in ('mae', 'rmse'):
                    v *= cfg.get('scale', 1.0)
                v = v.tolist()
                if isinstance(v, list):
                    for ii, vv in enumerate(v):
                        ret[f'{abbr}_{k}_{ii}'] = vv
                else:
                    ret[f'{abbr}_{k}'] = v
        if len(self.loss):
            for k, loss in self.loss.items():
                if not k.endswith('loss'):
                    k = k + '_loss'
                ret[k] = loss / self.samples

        logging.info(str(ret))

        return ret
