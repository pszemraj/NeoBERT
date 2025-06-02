import hydra
from omegaconf import DictConfig

from neobert.pretraining import trainer


@hydra.main(version_base=None, config_path="../../conf", config_name="pretraining")
def pipeline(cfg: DictConfig):
    trainer(cfg)


if __name__ == "__main__":
    pipeline()
