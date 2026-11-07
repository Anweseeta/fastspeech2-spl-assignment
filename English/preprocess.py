import argparse

import yaml

from preprocessor.preprocessor import Preprocessor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # CHANGED (SPL assignment): reads the combined config.yaml and uses its
    # `preprocess:` section (stock code expected a standalone preprocess.yaml).
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="path to combined config.yaml"
    )
    args = parser.parse_args()

    full_config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    config = full_config["preprocess"]
    preprocessor = Preprocessor(config)
    preprocessor.build_from_path()
