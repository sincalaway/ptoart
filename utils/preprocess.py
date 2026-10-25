from collections import namedtuple

import numpy as np

# ``rgba`` is the aligned square crop at native resolution; ``matrix`` is the
# 2x3 affine transform mapping original-photo pixels to crop pixels.
FaceRegion = namedtuple('FaceRegion', ['rgba', 'matrix'])


class Preprocess:
    def __init__(self, device='cpu', detector='dlib', seg_model_path=None):
        # Delay inference-heavy imports until an instance is constructed.
        from .face_detect import FaceDetect
        from .face_seg import FaceSeg

        self.detect = FaceDetect(device, detector)  # device = 'cpu' or 'cuda', detector = 'dlib' or 'sfd'
        self.segment = FaceSeg() if seg_model_path is None else FaceSeg(seg_model_path)

    def close(self):
        """Release the segmentation model's TensorFlow session, if initialized."""
        segment = getattr(self, 'segment', None)
        if segment is not None:
            segment.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def process(self, image):
        face_info = self.detect.align(image)
        if face_info is None:
            return None
        image_align, landmarks_align = face_info

        face, _, _ = self.__crop(image_align, landmarks_align)
        mask = self.segment.get_mask(face)
        return np.dstack((face, mask))

    def process_all(self, image, max_faces=10):
        """Return a FaceRegion for every detected face, largest first."""
        regions = []
        for image_align, landmarks_align, matrix in self.detect.align_all(image, max_faces=max_faces):
            face, left, top = self.__crop(image_align, landmarks_align)
            mask = self.segment.get_mask(face)
            to_crop = np.asarray(matrix, dtype=np.float64).copy()
            to_crop[0, 2] -= left
            to_crop[1, 2] -= top
            regions.append(FaceRegion(np.dstack((face, mask)), to_crop))
        return regions

    @staticmethod
    def __crop(image, landmarks):
        landmarks_top = np.min(landmarks[:, 1])
        landmarks_bottom = np.max(landmarks[:, 1])
        landmarks_left = np.min(landmarks[:, 0])
        landmarks_right = np.max(landmarks[:, 0])

        # expand bbox
        top = int(landmarks_top - 0.8 * (landmarks_bottom - landmarks_top))
        bottom = int(landmarks_bottom + 0.3 * (landmarks_bottom - landmarks_top))
        left = int(landmarks_left - 0.3 * (landmarks_right - landmarks_left))
        right = int(landmarks_right + 0.3 * (landmarks_right - landmarks_left))

        if bottom - top > right - left:
            left -= ((bottom - top) - (right - left)) // 2
            right = left + (bottom - top)
        else:
            top -= ((right - left) - (bottom - top)) // 2
            bottom = top + (right - left)

        crop_left, crop_top = left, top
        image_crop = np.ones((bottom - top + 1, right - left + 1, 3), np.uint8) * 255

        h, w = image.shape[:2]
        left_white = max(0, -left)
        left = max(0, left)
        right = min(right, w-1)
        right_white = left_white + (right-left)
        top_white = max(0, -top)
        top = max(0, top)
        bottom = min(bottom, h-1)
        bottom_white = top_white + (bottom - top)

        image_crop[top_white:bottom_white+1, left_white:right_white+1] = image[top:bottom+1, left:right+1].copy()
        return image_crop, crop_left, crop_top
