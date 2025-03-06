import numpy as np
from scipy import linalg
from tqdm import tqdm

import torch
from torch.nn.functional import interpolate

from pytorch_i3d_model.pytorch_i3d import InceptionI3d

NUMBER_OF_VIDEOS = 16
VIDEO_LENGTH = 15
PATH_TO_MODEL_WEIGHTS = "./pytorch_i3d_model/models/rgb_imagenet.pt"

def preprocess(videos, target_resolution):
    """
    videos: [N, T, H, W, C], range [0..255]
    Returns: [N, C, T, newH, newW], range [-1..1]
    """
    # Permute to [N, C, T, H, W]
    # e.g. if videos is [16, 15, 64, 64, 3], it becomes [16, 3, 15, 64, 64].
    reshaped_videos = videos.permute(0, 4, 1, 2, 3)

    # We want size=[T, newH, newW] for 'trilinear' interpolation.
    # e.g. [15, 224, 224]
    time_dim = reshaped_videos.size(2)  # T
    size = [time_dim] + list(target_resolution)

    # Resize in (C, T, H, W) space
    resized_videos = interpolate(
        reshaped_videos, 
        size=size, 
        mode='trilinear', 
        align_corners=False
    )

    # Scale from [0..255] to [-1..1]
    scaled_videos = 2.0 * resized_videos / 255.0 - 1.0
    return scaled_videos

def get_statistics(activations):
    """
    activations: [N, emb_dim] numpy array
    Returns: (mean, cov) for distribution
    """
    mean = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)  # [emb_dim, emb_dim]
    return mean, cov

def calculate_fvd_from_activations(first_activations, second_activations, eps=1e-10):
    """
    Compute Frechet Video Distance from two sets of embeddings.
    first_activations, second_activations: [N, emb_dim]
    """
    f_mean, f_cov = get_statistics(first_activations)
    s_mean, s_cov = get_statistics(second_activations)

    diff = f_mean - s_mean

    # sqrtm of product of covariances
    sqrt_cov = linalg.sqrtm(f_cov.dot(s_cov))
    if not np.isfinite(sqrt_cov).all():
        print("Sqrtm calculation produced singular values; adding %s to diagonals." % eps)
        offset = np.eye(f_cov.shape[0]) * eps
        sqrt_cov = linalg.sqrtm((f_cov + offset).dot(s_cov + offset))

    # Ensure real part (sometimes sqrtm returns complex arrays numerically)
    sqrt_cov = sqrt_cov.real

    return diff.dot(diff) + np.trace(f_cov + s_cov - 2.0 * sqrt_cov)

def get_activations(data, model, batch_size=10):
    """
    data: [N, C, T, H, W] Tensor
    model: PyTorch I3D
    Returns: Numpy array of shape [N, emb_dim]
    """
    all_activations = []

    # Break into batches along N dimension
    N = data.size(0)
    for start_idx in tqdm(range(0, N, batch_size), desc="Extracting activations"):
        end_idx = start_idx + batch_size
        batch = data[start_idx:end_idx]  # shape [batch_size, C, T, H, W]

        with torch.no_grad():
            # Model forward pass => shape [batch_size, channels, 1, 1, 1] or [batch_size, emb_dim]
            out = model(batch)

            # Flatten to [batch_size, emb_dim]
            if out.dim() > 2:
                out = out.view(out.size(0), -1)

        all_activations.append(out.cpu().numpy())

    # Stack => [N, emb_dim]
    return np.vstack(all_activations)

def frechet_video_distance(first_set_of_videos, second_set_of_videos, path_to_model_weights):
    # Load I3D
    i3d = InceptionI3d(400, in_channels=3)
    i3d.load_state_dict(torch.load(path_to_model_weights))
    i3d.eval()  # put model in eval mode

    # Preprocess each set to [N, C, T, newH, newW]
    first_processed = preprocess(first_set_of_videos, (224, 224))
    second_processed = preprocess(second_set_of_videos, (224, 224))

    print("Calculating activations for the first set of videos...")
    first_activations = get_activations(first_processed, i3d, batch_size=8)

    print("Calculating activations for the second set of videos...")
    second_activations = get_activations(second_processed, i3d, batch_size=8)

    fvd_score = calculate_fvd_from_activations(first_activations, second_activations)
    return fvd_score

def main():
    # Create dummy data: 16 "real" videos, 16 "fake" videos
    # Each: [N, T, H, W, C] => [16, 15, 64, 64, 3]
    first_set_of_videos = torch.zeros(
        NUMBER_OF_VIDEOS, VIDEO_LENGTH, 64, 64, 3
    )
    second_set_of_videos = torch.ones(
        NUMBER_OF_VIDEOS, VIDEO_LENGTH, 64, 64, 3
    ) * 255

    # Compute FVD
    fvd = frechet_video_distance(
        first_set_of_videos,
        second_set_of_videos,
        PATH_TO_MODEL_WEIGHTS
    )
    print("FVD:", fvd)

if __name__ == "__main__":
    main()