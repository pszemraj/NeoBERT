import hydra
from omegaconf import DictConfig

from neobert.glue import trainer


@hydra.main(version_base=None, config_path="../../conf", config_name="glue")
def run_glue(cfg: DictConfig):
    trainer(cfg)


if __name__ == "__main__":
    run_glue()
