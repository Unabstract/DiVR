import torch
import numpy as np
from scipy.spatial.transform import Rotation
# Compute the orthodromic distance between two points in 3d coordinates

def great_circle_distance(gt_quaternions, pred_quaternions):
    # Ensure the quaternions are normalized
    gt_quaternions /= torch.linalg.norm(gt_quaternions, dim=-1).unsqueeze(-1).repeat(1, 1, 4)
    pred_quaternions /= torch.linalg.norm(gt_quaternions, dim=-1).unsqueeze(-1).repeat(1, 1, 4)

    # Compute the dot product between ground truth and predicted quaternions
    dot_products = torch.sum(gt_quaternions * pred_quaternions, dim=-1)

    # Ensure dot products are within valid range [-1, 1]
    dot_products = torch.clamp(dot_products, min=-1, max=1)

    # Compute the great circle distance using arccos
    angular_distances = 2 * torch.acos(torch.abs(dot_products))

    return angular_distances

def rotation_vectors_to_quaternions(rotation_vectors):
    """Convert rotation vectors to quaternions."""
    quaternions = []
    for rotation_vector in rotation_vectors:
        rotation_matrices = Rotation.from_rotvec(rotation_vector.numpy()).as_matrix()
        quaternion = Rotation.from_matrix(rotation_matrices).as_quat()
        quaternions.append(quaternion)

    return torch.from_numpy(np.array(quaternions).copy()).float()

def compute_ade(predicted, ground_truth):
    """
    Compute the Average Displacement Error (ADE).

    Parameters:
    predicted (np.ndarray): Predicted positions tensor of shape (batch_size, number_pos, xyz).
    ground_truth (np.ndarray): Ground truth positions tensor of shape (batch_size, number_pos, xyz).

    Returns:
    float: The average displacement error across all predictions in the batch.
    """

    predicted = predicted.cpu().numpy()
    ground_truth = ground_truth.cpu().numpy()
    # Calculate the Euclidean distance for each point
    errors = np.linalg.norm(predicted - ground_truth, axis=-1)
    # Average over all positions and then average over all samples in the batch
    ade = np.mean(errors)
    return ade

def compute_fde(predicted, ground_truth):
    """
    Compute the Final Displacement Error (FDE).

    Parameters:
    predicted (np.ndarray): Predicted positions tensor of shape (batch_size, number_pos, xyz).
    ground_truth (np.ndarray): Ground truth positions tensor of shape (batch_size, number_pos, xyz).

    Returns:
    float: The final displacement error averaged across the batch.
    """
    # Extract the final positions
    final_predicted = predicted[:, -1, :].cpu().numpy()
    final_ground_truth = ground_truth[:, -1, :].cpu().numpy()
    # Calculate the Euclidean distance at the final position
    errors = np.linalg.norm(final_predicted - final_ground_truth, axis=-1)
    # Average the errors over all samples in the batch
    fde = np.mean(errors)
    return fde

