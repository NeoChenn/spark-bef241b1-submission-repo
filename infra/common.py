import modal

DATA_VOL = modal.Volume.from_name("rust-data", create_if_missing=True)
CKPT_VOL = modal.Volume.from_name("rust-ckpt", create_if_missing=True)
ASSET_VOL = modal.Volume.from_name("rig-assets", create_if_missing=True)

DATA_DIR = "/data"
CKPT_DIR = "/ckpt"
ASSET_DIR = "/assets"
