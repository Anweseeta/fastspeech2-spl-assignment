import os
import json

import torch
import numpy as np

import hifigan
from model import FastSpeech2, ScheduledOptim


def get_model(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = FastSpeech2(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        # CHANGED (SPL assignment): the vendored HiFi-GAN UNIVERSAL weights live
        # in the shared repo-root hifigan/ directory (too large to duplicate per
        # language). config.json is read from the imported hifigan package, and
        # the weights path comes from config["vocoder"]["weights_path"] with
        # fallbacks to the repo-root hifigan/ directory.
        hifigan_dir = os.path.dirname(os.path.abspath(hifigan.__file__))
        with open(os.path.join(hifigan_dir, "config.json"), "r") as f:
            hparams = json.load(f)
        hparams = hifigan.AttrDict(hparams)
        vocoder = hifigan.Generator(hparams)
        if speaker == "LJSpeech":
            ckpt_path = os.path.join(hifigan_dir, "generator_LJSpeech.pth.tar")
        elif speaker == "universal":
            candidates = [
                config["vocoder"].get("weights_path"),
                os.path.join(hifigan_dir, "generator_universal.pth.tar"),
                os.path.abspath(
                    os.path.join(hifigan_dir, "..", "..", "hifigan",
                                 "generator_universal.pth.tar")
                ),
            ]
            ckpt_path = next(
                (c for c in candidates if c and os.path.exists(c)), None
            )
            if ckpt_path is None:
                raise FileNotFoundError(
                    "HiFi-GAN universal weights not found. Tried: {}. "
                    "Place generator_universal.pth.tar in the repo-root hifigan/ "
                    "directory or set vocoder.weights_path in config.yaml.".format(
                        candidates
                    )
                )
        ckpt = torch.load(ckpt_path, map_location=device)
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
