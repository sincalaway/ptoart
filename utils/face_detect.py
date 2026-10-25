import cv2
import math
import numpy as np
import face_alignment


# Face detectors (SFD in particular) allocate feature maps proportional to the
# input area: a 20 MP phone photo needs >5 GB. Detect on a downscaled copy and
# map the boxes back, then find landmarks on the full-resolution image.
DETECT_MAX_SIDE = 1280


class FaceDetect:
    def __init__(self, device, detector):
        # landmarks will be detected by face_alignment library. Set device = 'cuda' if use GPU.
        landmarks_type = face_alignment.LandmarksType
        two_d = getattr(landmarks_type, 'TWO_D', None)
        if two_d is None:
            two_d = getattr(landmarks_type, '_2D')
        self.fa = face_alignment.FaceAlignment(two_d, device=device, face_detector=detector)

    def _landmarks(self, image):
        """Return face_alignment landmark predictions with bounded detector memory."""
        height, width = image.shape[:2]
        scale = min(1.0, DETECT_MAX_SIDE / max(height, width))
        if scale >= 1.0:
            return self.fa.get_landmarks(image)
        small = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                           interpolation=cv2.INTER_AREA)
        detector = getattr(self.fa, 'face_detector', None)
        if detector is not None and hasattr(detector, 'detect_from_image'):
            boxes = detector.detect_from_image(small.copy())
            if boxes is None or len(boxes) == 0:
                return None
            boxes = [np.concatenate([np.asarray(b[:4], dtype=np.float32) / scale,
                                     np.asarray(b[4:], dtype=np.float32)]) for b in boxes]
            return self.fa.get_landmarks(image, detected_faces=boxes)
        preds = self.fa.get_landmarks(small)
        if not preds:
            return preds
        return [np.asarray(pred, dtype=np.float32) / scale for pred in preds]

    def align(self, image):
        landmarks = self.__get_max_face_landmarks(image)

        if landmarks is None:
            return None

        else:
            return self.__rotate(image, landmarks)

    def align_all(self, image, max_faces=10, min_size=24):
        """Align every detected face, largest first.

        Returns a list of ``(rotated_image, rotated_landmarks, matrix)`` where
        ``matrix`` is the 2x3 affine transform from original to rotated pixels.
        Faces whose landmark box is narrower than ``min_size`` pixels are skipped.
        """
        preds = self._landmarks(image)
        if not preds:
            return []
        faces = []
        for pred in preds:
            width = np.ptp(pred[:, 0])
            if width < min_size:
                continue
            faces.append((width * np.ptp(pred[:, 1]), pred))
        faces.sort(key=lambda item: item[0], reverse=True)
        return [self.__rotate(image, pred, return_matrix=True) for _, pred in faces[:max_faces]]

    def __get_max_face_landmarks(self, image):
        preds = self._landmarks(image)
        if preds is None or len(preds) == 0:
            return None

        elif len(preds) == 1:
            return preds[0]

        else:
            # find max face
            areas = []
            for pred in preds:
                landmarks_top = np.min(pred[:, 1])
                landmarks_bottom = np.max(pred[:, 1])
                landmarks_left = np.min(pred[:, 0])
                landmarks_right = np.max(pred[:, 0])
                areas.append((landmarks_bottom - landmarks_top) * (landmarks_right - landmarks_left))
            max_face_index = np.argmax(areas)
            return preds[max_face_index]

    @staticmethod
    def __rotate(image, landmarks, return_matrix=False):
        # rotation angle
        left_eye_corner = landmarks[36]
        right_eye_corner = landmarks[45]
        eye_delta_x = left_eye_corner[0] - right_eye_corner[0]
        eye_delta_y = left_eye_corner[1] - right_eye_corner[1]
        # Coincident horizontal coordinates do not define a stable roll angle;
        # leave the image unrotated rather than dividing by zero.
        radian = 0.0 if eye_delta_x == 0 else np.arctan(eye_delta_y / eye_delta_x)

        # image size after rotating
        height, width, _ = image.shape
        cos = math.cos(radian)
        sin = math.sin(radian)
        new_w = int(width * abs(cos) + height * abs(sin))
        new_h = int(width * abs(sin) + height * abs(cos))

        # translation
        Tx = new_w // 2 - width // 2
        Ty = new_h // 2 - height // 2

        # affine matrix
        M = np.array([[cos, sin, (1 - cos) * width / 2. - sin * height / 2. + Tx],
                      [-sin, cos, sin * width / 2. + (1 - cos) * height / 2. + Ty]])

        image_rotate = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(255, 255, 255))

        landmarks = np.concatenate([landmarks, np.ones((landmarks.shape[0], 1))], axis=1)
        landmarks_rotate = np.dot(M, landmarks.T).T
        if return_matrix:
            return image_rotate, landmarks_rotate, M
        return image_rotate, landmarks_rotate


if __name__ == '__main__':
    img = cv2.cvtColor(cv2.imread('3989161_1.jpg'), cv2.COLOR_BGR2RGB)
    fd = FaceDetect(device='cpu')
    face_info = fd.align(img)
    if face_info is not None:
        image_align, landmarks_align = face_info

        for i in range(landmarks_align.shape[0]):
            cv2.circle(image_align, (int(landmarks_align[i][0]), int(landmarks_align[i][1])), 2, (255, 0, 0), -1)

        cv2.imwrite('image_align.png', cv2.cvtColor(image_align, cv2.COLOR_RGB2BGR))
