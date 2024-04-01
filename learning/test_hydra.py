import hydra
from omegaconf import OmegaConf

# loading
config = OmegaConf.load("config.yaml")


@hydra.main(config_name="config.yaml", config_path=".", version_base="1.1")
def main(cfg):
    # Print the config file using `to_yaml` method which prints in a pretty manner
    print(OmegaConf.to_yaml(cfg))
    print(cfg.preferences.user)
    print(cfg.preferences.trait)


if __name__ == "__main__":
    main()
