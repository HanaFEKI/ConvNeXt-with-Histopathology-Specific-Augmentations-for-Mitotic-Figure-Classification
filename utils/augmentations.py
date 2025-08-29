import cv2
import numpy as np
import albumentations as A

class ElasticTransformHistopath(A.DualTransform):
    """Elastic transformation optimized for histopathology images"""
    def __init__(self, alpha=50, sigma=5, alpha_affine=10, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.alpha = alpha
        self.sigma = sigma
        self.alpha_affine = alpha_affine

    def apply(self, img, random_state=None, **params):
        return self.elastic_transform(img, self.alpha, self.sigma, self.alpha_affine, random_state)

    def apply_to_mask(self, img, random_state=None, **params):
        return self.elastic_transform(img, self.alpha, self.sigma, self.alpha_affine, random_state)

    def elastic_transform(self, image, alpha, sigma, alpha_affine, random_state):
        if random_state is None:
            random_state = np.random.RandomState(None)
        shape = image.shape
        shape_size = shape[:2]

        # Random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32([center_square + square_size,
                          [center_square[0] + square_size, center_square[1] - square_size],
                          center_square - square_size])
        pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

        # Elastic transformation
        blur_size = int(4 * sigma) | 1
        dx = cv2.GaussianBlur((random_state.rand(*shape_size) * 2 - 1).astype(np.float32),
                             ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
        dy = cv2.GaussianBlur((random_state.rand(*shape_size) * 2 - 1).astype(np.float32),
                             ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
        x, y = np.meshgrid(np.arange(shape_size[1]), np.arange(shape_size[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        return cv2.remap(image, indices[1], indices[0], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    def get_transform_init_args_names(self):
        return ("alpha", "sigma", "alpha_affine")
