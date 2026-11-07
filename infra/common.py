"""Shared Modal names. Every Modal app in this repo imports from here so volumes line up."""
import modal

DATA_VOL = modal.Volume.from_name("rust-data", create_if_missing=True)    # /data   : detection dataset
CKPT_VOL = modal.Volume.from_name("rust-ckpt", create_if_missing=True)    # /ckpt   : model + policy checkpoints
ASSET_VOL = modal.Volume.from_name("rig-assets", create_if_missing=True)  # /assets : URDF / USD from the sim team

DATA_DIR = "/data"
CKPT_DIR = "/ckpt"
ASSET_DIR = "/assets"
