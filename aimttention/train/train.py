import click
import omegaconf
from omegaconf import OmegaConf
from aimttention.config import build_module
import torch
from ignite import distributed as idist
import logging
from aimttention.modules import Forces
from aimttention.train import utils


@click.command()
@click.option('--config', type=click.Path(exists=True), required=True,
    help='Path to training configuration YAML file.')
@click.option('--model', type=click.Path(exists=True), required=True,
    help='Path to the model definition YAML file.')
@click.option('--load', type=click.Path(exists=True), default=None,
    help='Path to the model weights to load.')
@click.option('--save', type=click.Path(), default=None,
    help='Path to save the model weights.')
@click.argument('args', type=str, nargs=-1)
def train(config, model, load, save, args):
    """Train AIMNet2 model.
    ARGS are one or more parameters to overwrite in config in a dot-separated form.
    For example: `data.train=mydataset.h5`.
    """
    logging.basicConfig(level=logging.INFO)

    # model config
    logging.info('Start training')
    logging.info(f'Using model definition: {model}')
    model_cfg = OmegaConf.load(model)
    logging.info('--- START model.yaml ---')
    model_cfg = OmegaConf.to_yaml(model_cfg)
    logging.info(model_cfg)
    logging.info('--- END model.yaml ---')

    # train config
    logging.info(f'Using training configuration: {config}')
    train_cfg = OmegaConf.load(config)
    if args:
        logging.info('Overriding configuration:')
        for arg in args:
            logging.info(arg)
        args_cfg = OmegaConf.from_dotlist(args)
        train_cfg = OmegaConf.merge(train_cfg, args_cfg)
    logging.info('--- START train.yaml ---')
    train_cfg = OmegaConf.to_yaml(train_cfg)
    logging.info(train_cfg)
    logging.info('--- END train.yaml ---')

    # launch — CUDA or CPU only (MPS not supported)
    num_gpus = torch.cuda.device_count()
    logging.info(f'Found {num_gpus} CUDA GPU(s):')
    for i in range(num_gpus):
        logging.info(f'  [{i}] {torch.cuda.get_device_name(i)}')
    if num_gpus == 0:
        logging.warning('No CUDA GPU available. Training will run on CPU.')
        run(0, model_cfg, train_cfg, load, save)
    elif num_gpus == 1:
        logging.info('Single-GPU training.')
        run(0, model_cfg, train_cfg, load, save)
    else:
        logging.info(f'DDP training on {num_gpus} GPUs.')
        with idist.Parallel(backend='nccl', nproc_per_node=num_gpus) as parallel:
            parallel.run(run, model_cfg, train_cfg, load, save)


def run(local_rank, model_cfg, train_cfg, load, save):
    if local_rank == 0:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.ERROR)

    # load configs
    model_cfg = OmegaConf.create(model_cfg)
    train_cfg = OmegaConf.create(train_cfg)

    # device: CUDA or CPU (no MPS)
    ddp = idist.get_world_size() > 1
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    # auto-detect num_species from SAE file
    if train_cfg.data.get('sae') is not None:
        from aimttention.config import load_yaml
        max_z = 0
        for k, c in train_cfg.data.sae.items():
            sae = load_yaml(c.file)
            max_z = max(max_z, max(sae.keys()))
        num_species = max_z + 1
        with omegaconf.open_dict(model_cfg):
            model_cfg.kwargs.num_species = num_species
        logging.info(f'Auto-detected num_species={num_species} from SAE file')

    # build model
    _force_training = 'forces' in train_cfg.data.y
    model = utils.build_model(model_cfg, forces=_force_training)
    model = utils.set_trainable_parameters(model,
            train_cfg.optimizer.force_train,
            train_cfg.optimizer.force_no_train)
    model = model.to(device)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank)

    # load weights
    if load is not None:
        logging.info(f'Loading weights from file {load}')
        sd = torch.load(load, map_location=device)
        logging.info(utils.unwrap_module(model).load_state_dict(sd, strict=False))

    # data loaders
    train_loader, val_loader = utils.get_loaders(train_cfg.data)

    # optimizer
    optimizer = utils.get_optimizer(model, train_cfg.optimizer)

    # scheduler
    if train_cfg.scheduler is not None:
        scheduler = utils.get_scheduler(optimizer, train_cfg.scheduler)
    else:
        scheduler = None

    loss = utils.get_loss(train_cfg.loss)
    metrics = utils.get_metrics(train_cfg.metrics)
    metrics.attach_loss(loss)

    # ignite engine
    trainer, validator = utils.build_engine(
        model, optimizer, scheduler, loss, metrics, train_cfg, val_loader)

    _use_wandb = local_rank == 0 and train_cfg.wandb is not None
    if _use_wandb:
        utils.setup_wandb(train_cfg, model_cfg, model, trainer, validator, optimizer)

    trainer.run(train_loader, max_epochs=train_cfg.trainer.epochs)

    # sync all ranks before saving
    if ddp:
        torch.distributed.barrier()

    if local_rank == 0 and save is not None:
        logging.info(f'Saving model weights to file {save}')
        torch.save(utils.unwrap_module(model).state_dict(), save)

    if _use_wandb:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    train()
