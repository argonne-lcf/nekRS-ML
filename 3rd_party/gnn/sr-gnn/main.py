import os
import hydra
from omegaconf import DictConfig, OmegaConf
import time

from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
LOCAL_RANK = int(os.getenv("PALS_LOCAL_RANKID"))
LOCAL_SIZE = int(os.getenv("PALS_LOCAL_SIZE"))
HOST_NAME = MPI.Get_processor_name()

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


def train(cfg: DictConfig):
    start = time.time()
    from trainer import Trainer # needed here to avoid clash with hydra config parsing
    trainer = Trainer(cfg, COMM)
    epoch_times = []
    valid_times = []

    for epoch in range(trainer.epoch_start, cfg.epochs + 1):
        # ~~~~ Training step
        t0 = time.time()
        trainer.epoch = epoch
        train_metrics = trainer.train_epoch(epoch)
        trainer.loss_hist_train[epoch - 1] = train_metrics["loss"]
        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)

        # ~~~~ Validation step
        t0 = time.time()
        test_metrics = trainer.test()
        trainer.loss_hist_test[epoch - 1] = test_metrics["loss"]
        valid_time = time.time() - t0
        valid_times.append(valid_time)

        # ~~~~ Learning rate
        lr = trainer.optimizer.param_groups[0]["lr"]
        trainer.lr_hist[epoch - 1] = lr

        if RANK == 0:
            summary = "  ".join([
                "[TRAIN]",
                f"loss={train_metrics['loss']:.4e}",
                f"epoch_time={epoch_time:.4g} sec",
                f" valid_time={valid_time:.4g} sec",
                f" learning_rate={lr:.6g}",
            ])
            logger.info((sep := "-" * len(summary)))
            logger.info(summary)
            logger.info(sep)
            astr = f"[VALIDATION] loss={test_metrics['loss']:.4e}"
            sepstr = "-" * len(astr)
            logger.info(sepstr)
            logger.info(astr)
            logger.info(sepstr)

        # ~~~~ Step scheduler based on validation loss
        trainer.scheduler.step(test_metrics["loss"])

        # ~~~~ Checkpointing step
        if epoch % cfg.ckptfreq == 0:
            trainer.checkpoint(epoch)

    rstr = f"[{RANK}] ::"
    if RANK == 0:
        logger.info(
            " ".join([
                rstr,
                f"Total training time: {time.time() - start} seconds",
            ])
        )

    trainer.save_model()
    trainer.cleanup()


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if RANK == 0:
        logger.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        logger.info("INPUTS:")
        logger.info(OmegaConf.to_yaml(cfg))
        logger.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    train(cfg)


if __name__ == "__main__":
    main()
