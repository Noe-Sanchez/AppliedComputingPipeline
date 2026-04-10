import cv2
import numpy
import torch
import numpy as np

from depth_anything_v2.dpt import DepthAnythingV2


h, w = 1080, 1920
img_center = np.array([w / 2, h / 2])

print(torch.cuda.is_available())
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitl' # or 'vits', 'vitb', 'vitl', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()


raw_img = cv2.imread('frame_300.jpg')


def segment_trees(img = numpy.ndarray):
    try:
        frame = img.copy()

        blur = cv2.blur(frame, (20, 20))
        imgHsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        lower = np.array([20, 18, 0])
        upper = np.array([76, 255, 255])
        color_mask = cv2.inRange(imgHsv, lower, upper)

        depth = model.infer_image(frame)
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_norm = depth_norm.astype('uint8')
        _, depth_mask = cv2.threshold(depth_norm, 100, 255, cv2.THRESH_BINARY)

        both = cv2.bitwise_and(depth_mask, color_mask)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(both, connectivity=8)

        min_area = 200
        best_label = None
        best_dist = float('inf')

        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < min_area:
                continue

            cx, cy = centroids[label]
            dist = np.linalg.norm(np.array([cx, cy]) - img_center)

            if dist < best_dist:
                best_dist = dist
                best_label = label

        center_mask = np.zeros_like(both)
        if best_label is not None:
            center_mask[labels == best_label] = 255

        return center_mask

    except:
        print("error")

