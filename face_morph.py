import dlib
import numpy as np
import cv2
from PIL import Image
from moviepy.editor import VideoClip, concatenate_videoclips
from scipy.spatial import Delaunay

from Lib import os

DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR_PATH = "./shape_predictor_68_face_landmarks.dat"
PREDICTOR = dlib.shape_predictor(PREDICTOR_PATH)


class MorphVideoClip(VideoClip):
    def __init__(self, im1, im2, duration, im1_landmarks, im2_landmarks, triangulation):
        self.im1 = im1
        self.im2 = im2
        self.duration = duration
        self.im1_landmarks = im1_landmarks
        self.im2_landmarks = im2_landmarks
        self.triangulation = triangulation
        self.size = im1.shape[1], im1.shape[0]  # Breite und Höhe des Bildes im1
        super().__init__(duration=duration)

    # Überschreiben Sie die make_frame Methode
    def make_frame(self, t):
        return self.morph_frame(t)

    def morph_frame(self, t):
        alpha = t / self.duration
        im1 = np.float32(self.im1)
        im2 = np.float32(self.im2)

        weighted_landmarks = (1 - alpha) * self.im1_landmarks + alpha * self.im2_landmarks
        warped_im1 = self.warp_im(im1, self.im1_landmarks, weighted_landmarks)
        warped_im2 = self.warp_im(im2, self.im2_landmarks, weighted_landmarks)

        morphed_im = cv2.addWeighted(warped_im1, 1 - alpha, warped_im2, alpha, 0)
        morphed_im = cv2.cvtColor(morphed_im, cv2.COLOR_BGR2RGB)

        return morphed_im

    def warp_im(self, im, src_landmarks, dst_landmarks):
        im_out = im.copy()

        for i in range(len(self.triangulation)):
            src_tri = src_landmarks[self.triangulation[i]]
            dst_tri = dst_landmarks[self.triangulation[i]]
            self.morph_triangle(im, im_out, src_tri, dst_tri)

        return im_out

    @staticmethod
    def morph_triangle(im, im_out, src_tri, dst_tri):
        sr = cv2.boundingRect(np.float32([src_tri]))
        dr = cv2.boundingRect(np.float32([dst_tri]))

        cropped_src_tri = [(src_tri[i][0] - sr[0], src_tri[i][1] - sr[1]) for i in range(3)]
        cropped_dst_tri = [(dst_tri[i][0] - dr[0], dst_tri[i][1] - dr[1]) for i in range(3)]

        mask = np.zeros((dr[3], dr[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(cropped_dst_tri), (1.0, 1.0, 1.0), 16, 0)

        cropped_im = im[sr[1]:sr[1] + sr[3], sr[0]:sr[0] + sr[2]]

        size = (dr[2], dr[3])
        warp_image = MorphVideoClip.affine_transform(cropped_im, cropped_src_tri, cropped_dst_tri, size)

        im_out[dr[1]:dr[1] + dr[3], dr[0]:dr[0] + dr[2]] = \
            im_out[dr[1]:dr[1] + dr[3], dr[0]:dr[0] + dr[2]] * (1 - mask) + warp_image * mask

    @staticmethod
    def affine_transform(src, src_tri, dst_tri, size):
        M = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
        dst = cv2.warpAffine(src, M, size, borderMode=cv2.BORDER_REFLECT_101)
        return dst

    @staticmethod
    def prompt_user_to_choose_face(image, rects):
        # Simple Auswahl des ersten erkannten Gesichts
        if not rects:
            raise ValueError("No face rectangles found in the image.")

        chosen_rect = rects[0]  # Das erste erkannte Gesicht auswählen
        return chosen_rect
    @staticmethod
    def get_boundary_points(image_shape):
        h, w = image_shape[:2]
        boundary_points = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        return boundary_points


def calculate_delaunay_triangles(points):
    tri = Delaunay(points)
    return tri.simplices


def calculate_landmarks(image):
    rects = DETECTOR(image, 1)
    if len(rects) == 0 and len(DETECTOR(image, 0)) > 0:
        rects = DETECTOR(image, 0)

    if len(rects) == 0:
        return None, None

    target_rect = rects[0]
    if len(rects) > 1:
        target_rect = MorphVideoClip.prompt_user_to_choose_face(image, rects)

    landmarks = np.array([(p.x, p.y) for p in PREDICTOR(image, target_rect).parts()])
    landmarks = np.append(landmarks, MorphVideoClip.get_boundary_points(image.shape), axis=0)

    triangulation = calculate_delaunay_triangles(landmarks)

    return landmarks, triangulation


def morph_images(im_dir, im_files, duration, fps, pause_duration, out_name):
    clips = []

    im1 = cv2.imread(os.path.join(im_dir, im_files[0]))
    im1_landmarks, triangulation = calculate_landmarks(im1)

    for i in range(len(im_files) - 1):
        im2 = cv2.imread(os.path.join(im_dir, im_files[i + 1]))
        im2_landmarks, _ = calculate_landmarks(im2)

        if im1_landmarks is None or im2_landmarks is None:
            print("No faces found, skipping morphing between images.")
            continue

        clip = MorphVideoClip(im1, im2, duration, im1_landmarks, im2_landmarks, triangulation)
        clip = clip.set_fps(fps)
        clips.append(clip)

        if pause_duration > 0 and i < len(im_files) - 2:
            pause_clip = VideoClip(lambda t: np.zeros((1, 1, 3), dtype=np.uint8)).set_duration(pause_duration)
            clips.append(pause_clip)

        im1 = im2
        im1_landmarks = im2_landmarks

    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(out_name, codec='libx264', fps=fps)
