from data import load_data
import config
import argparse
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from modules.denoising_diffusion import GaussianDiffusion
from modules.unet import Unet
from modules.trainer import Trainer
from modules.temporal_models import HistoryNet, CondNet
from torch.nn.parallel import DistributedDataParallel as DDP
import os


parser = argparse.ArgumentParser(description="values from bash script")
parser.add_argument("--ndevice", type=int, required=True, help="number of cuda device")
args = parser.parse_args()


def schedule_func(ep):
    return max(config.decay ** ep, config.minf)


def main(rank, world_size):
<<<<<<< HEAD
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
=======
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
>>>>>>> 4edb935 (commented)

    dist.init_process_group("nccl", rank=rank, world_size=world_size)   # this allows processes to communicate with each other

    train_data, val_data = load_data(
        config.data_config,
        config.batch_size,
        pin_memory=True,
        num_workers=4,
    )

    denoise_model = Unet(
        dim=config.embed_dim,
        context_dim_factor=config.context_dim_factor,
        channels=config.data_config["img_channel"],
        dim_mults=config.dim_mults,
    )

    context_model = CondNet(
        dim=int(config.context_dim_factor * config.embed_dim),
        channels=config.data_config["img_channel"],
        backbone=config.backbone,
        dim_mults=config.dim_mults,
    )

    transform_model = (
        HistoryNet(
            dim=int(config.transform_dim_factor * config.embed_dim),
            channels=config.data_config["img_channel"],
            context_mode=config.transform_mode,
            dim_mults=config.transform_dim_mults,
            backbone=config.backbone
        )
        if config.transform_mode
        in ["residual"]
        else None
    )

    diffusion = GaussianDiffusion(
        denoise_fn=denoise_model,  # Unet
        history_fn=context_model,  # CondNet
        transform_fn=transform_model,  # (HistoryNet)
        pred_mode=config.pred_mode,  # noise
        clip_noise=config.clip_noise,  # True
        timesteps=config.iteration_step,  # 1600
        loss_type=config.loss_type,  # l1
        aux_loss=False,
    ).to(rank)

    diffusion = DDP(diffusion, device_ids=[rank])  # wrap GaussianDiffusion inside a DistributedDataParallel

    trainer = Trainer(
        rank=rank,  # used for multiprocessing
        diffusion_model=diffusion,  # GaussianDiffusion
        train_dl=train_data,
        val_dl=val_data,
        sample_num_of_frame=config.init_num_of_frame + 1,
        init_num_of_frame=config.init_num_of_frame,
        scheduler_function=schedule_func,
        scheduler_checkpoint_step=config.scheduler_checkpoint_step,
        ema_decay=config.ema_decay,
        train_lr=config.lr,
        train_num_steps=config.n_step,
        step_start_ema=config.ema_start_step,
        update_ema_every=config.ema_step,
        save_and_sample_every=config.log_checkpoint_step,
        results_folder=os.path.join(config.result_root, f"{config.model_name}/"),
        tensorboard_dir=os.path.join(config.tensorboard_root, f"{config.model_name}/"),
        model_name=config.model_name,
        val_num_of_batch=config.val_num_of_batch,
        optimizer=config.optimizer
    )

    if config.load_model:
        trainer.load(load_step=config.load_step)

    trainer.train()


if __name__ == "__main__":
    mp.spawn(main, args=(args.ndevice,), nprocs=args.ndevice, join=True)
    dist.barrier()  # Synchronizes all processes. This collective blocks processes until the whole group enters this function,
                    # if async_op is False (the default), or if async work handle is called on wait().
    dist.destroy_process_group()
