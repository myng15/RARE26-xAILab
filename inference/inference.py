from pathlib import Path
import json
from glob import glob

import numpy as np
import SimpleITK

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")


def run():
    interface_key = get_interface_key()
    handler = {("stacked-barretts-esophagus-endoscopy-images",): interface_0_handler}[interface_key]
    return handler()


def interface_0_handler():
    stacked = load_image_file_as_array(
        location=INPUT_PATH / "images/stacked-barretts-esophagus-endoscopy")
    frames = list(as_frames(stacked))
    print(f"Loaded stacked input: {stacked.shape} -> {len(frames)} frame(s)")
    _show_torch_cuda_info()
    likelihoods = predict_neoplasia(frames)
    write_json_file(
        location=OUTPUT_PATH / "stacked-neoplastic-lesion-likelihoods.json",
        content=likelihoods)
    return 0


def predict_neoplasia(frames):
    """Score each frame with the DINOv2 attention-pooling ensemble (mean of per-frame sigmoids).

    Scoring is per-frame independent, so results do not depend on how the input stack is batched."""
    from model.dinov2_classifier import DINOv2AttentionEnsemble

    ckpts = sorted(glob(str(RESOURCE_PATH / "dinov2_lora_attnmil_neoplasia_seed*.pt")))
    if not ckpts:
        raise FileNotFoundError("no dinov2_lora_attnmil_neoplasia_seed*.pt checkpoints in resources/")
    clf = DINOv2AttentionEnsemble(ckpts)
    scores = clf.predict(frames)
    print(f"DINOv2 attention-pooling ensemble ({len(ckpts)} seeds): scored {len(scores)} frame(s)")
    return [float(p) for p in scores]


def as_frames(stacked: np.ndarray):
    arr = np.asarray(stacked)
    if arr.ndim == 4:
        for i in range(arr.shape[0]):
            yield arr[i]
    elif arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
        yield arr
    elif arr.ndim == 2:
        yield arr
    else:
        raise ValueError(f"Unexpected stacked-image shape: {arr.shape}")


def get_interface_key():
    inputs = load_json_file(location=INPUT_PATH / "inputs.json")
    return tuple(sorted(sv["interface"]["slug"] for sv in inputs))


def load_json_file(*, location):
    with open(location, "r") as f:
        return json.loads(f.read())


def write_json_file(*, location, content):
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


def load_image_file_as_array(*, location):
    input_files = (glob(str(location / "*.tif")) + glob(str(location / "*.tiff"))
                   + glob(str(location / "*.mha")))
    result = SimpleITK.ReadImage(input_files[0])
    return SimpleITK.GetArrayFromImage(result)


def _show_torch_cuda_info():
    import torch
    print("=+=" * 10)
    print(f"Torch CUDA available: {(a := torch.cuda.is_available())}")
    if a:
        print(f"\tdevice: {torch.cuda.get_device_properties(torch.cuda.current_device())}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
