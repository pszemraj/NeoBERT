import hydra
from omegaconf import DictConfig

from neobert.contrastive import trainer


@hydra.main(version_base=None, config_path="../../conf", config_name="finetuning")
def pipeline(cfg: DictConfig):
    trainer(cfg)


if __name__ == "__main__":
    pipeline()
