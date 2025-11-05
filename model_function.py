import torch
import os

from sklearn.cluster import KMeans

categories = os.listdir('./dataset/CNN letter Dataset')

def license_plate_to_text(image_plate: torch.Tensor, characters_labels):
    result = []
    num_chars = image_plate.shape[0]
    if not (8 <= num_chars <= 9):
        return 'Unknown'

    x_centers = ((image_plate[:, 0] + image_plate[:, 2]) / 2).numpy()
    y_centers = ((image_plate[:, 1] + image_plate[:, 3]) / 2).numpy()
    heights = (image_plate[:, 3] - image_plate[:, 1]).numpy()

    kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(y_centers.reshape(-1, 1))
    labels = kmeans.labels_

    y_dist = abs(kmeans.cluster_centers_[0][0] - kmeans.cluster_centers_[1][0])
    avg_height = heights.mean()

    if y_dist < avg_height * 0.5:
        sorted_indices = sorted(range(num_chars), key=lambda i: x_centers[i])
        for idx in sorted_indices:
            result.append((str(characters_labels[idx])))
    else:
        cluster_info = [(i, kmeans.cluster_centers_[i][0]) for i in range(2)]
        cluster_info.sort(key=lambda x: x[1])
        top_cluster = cluster_info[0][0]
        bottom_cluster = cluster_info[1][0]

        top_indices = [i for i in range(num_chars) if labels[i] == top_cluster]
        top_indices.sort(key=lambda i: x_centers[i])
        for idx in top_indices:
            result.append((characters_labels[idx]))

        bottom_indices = [i for i in range(num_chars) if labels[i] == bottom_cluster]
        bottom_indices.sort(key=lambda i: x_centers[i])
        for idx in bottom_indices:
            result.append((characters_labels[idx]))

    return result
