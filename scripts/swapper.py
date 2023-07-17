import copy
import math
import os
import tempfile
from dataclasses import dataclass
from typing import List, Union, Dict, Set, Tuple

import cv2
import numpy as np
from PIL import Image

import insightface
import onnxruntime
from scripts.cimage import convert_to_sd

from modules.face_restoration import FaceRestoration, restore_faces
from modules.upscaler import Upscaler, UpscalerData
from scripts.roop_logging import logger

providers = onnxruntime.get_available_providers()


@dataclass
class UpscaleOptions:
    scale: int = 1
    upscaler: UpscalerData = None
    upscale_visibility: float = 0.5
    face_restorer: FaceRestoration = None
    restorer_visibility: float = 0.5

FS_MODEL = None
CURRENT_FS_MODEL_PATH = None


def getFaceSwapModel(model_path: str):
    global FS_MODEL
    global CURRENT_FS_MODEL_PATH
    if CURRENT_FS_MODEL_PATH is None or CURRENT_FS_MODEL_PATH != model_path:
        CURRENT_FS_MODEL_PATH = model_path
        FS_MODEL = insightface.model_zoo.get_model(model_path, providers=providers)

    return FS_MODEL


def upscale_image(image: Image, upscale_options: UpscaleOptions):
    result_image = image
    if upscale_options.upscaler is not None and upscale_options.upscaler.name != "None":
        original_image = result_image.copy()
        logger.info(
            "Upscale with %s scale = %s",
            upscale_options.upscaler.name,
            upscale_options.scale,
        )
        result_image = upscale_options.upscaler.scaler.upscale(
            image, upscale_options.scale, upscale_options.upscaler.data_path
        )
        if upscale_options.scale == 1:
            result_image = Image.blend(
                original_image, result_image, upscale_options.upscale_visibility
            )

    if upscale_options.face_restorer is not None:
        original_image = result_image.copy()
        logger.info("Restore face with %s", upscale_options.face_restorer.name())
        numpy_image = np.array(result_image)
        numpy_image = upscale_options.face_restorer.restore(numpy_image)
        restored_image = Image.fromarray(numpy_image)
        result_image = Image.blend(
            original_image, restored_image, upscale_options.restorer_visibility
        )

    return result_image


def get_face_single(img_data: np.ndarray, face_index=0, det_size=(640, 640), det_thresh = 0.8, gender = 0):
    face_analyser = insightface.app.FaceAnalysis(name="buffalo_l", providers=providers)
    face_analyser.prepare(ctx_id=0, det_thresh=det_thresh, det_size=det_size)
    faces = face_analyser.get(img_data)

    # if there are no faces detected, try again with a smaller det_size
    if len(faces) == 0 and det_size[0] > 320 and det_size[1] > 320:
        det_size_half = (det_size[0] // 2, det_size[1] // 2)
        return get_face_single(img_data, face_index=face_index, det_size=det_size_half)
    
    if len(faces) == 0 and det_thresh > 0.1:
        return get_face_single(img_data, face_index=face_index, det_size=det_size, det_thresh=det_thresh-0.1)
    
    #print(faces)
    if len(faces) <= 0:
        return None
    
    # print out all the face.gender, face.sex and face.age values
    for face in faces:
        print(f"gender={face.gender}, sex={face.sex}, age={face.age}");

    # filter out faces that aren't the indicated gender using face.gender
    gender_faces = list(filter(lambda face: True if face.gender == gender else False, faces))

    # if we couldn't find gender, use all faces
    if len(faces) == 0:
        gender_faces = faces

    # Calculate the center of the image
    image_height, image_width = img_data.shape[:2]
    image_center = (image_width // 2, image_height // 2)

    # Calculate areas and categorize bounding boxes
    areas = [(face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]) for face in gender_faces]

    # Using histogram to categorize into 3 bins
    _, bin_edges = np.histogram(areas, bins=3)

    bins = {'small': [], 'medium': [], 'large': []}
    for face in gender_faces:
        area = (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
        if area < bin_edges[1]:
            bins['small'].append(face)
        elif area < bin_edges[2]:
            bins['medium'].append(face)
        else:
            bins['large'].append(face)

    # Sort within each bin by distance from center
    sorted_bins = {}
    for size in ['large', 'medium', 'small']:
        sorted_bins[size] = sorted(
            bins[size], key=lambda face: ((face.bbox[2] + face.bbox[0]) / 2 - image_center[0]) ** 2 + ((face.bbox[3] + face.bbox[1]) / 2 - image_center[1]) ** 2
        )

    # Flatten the sorted bins into a single list
    flattened_sorted_bins = [face for size in ['large', 'medium', 'small'] for face in sorted_bins[size]]

    return flattened_sorted_bins[face_index]


@dataclass
class ImageResult:
    path: Union[str, None] = None
    similarity: Union[Dict[int, float], None] = None  # face, 0..1

    def image(self) -> Union[Image.Image, None]:
        if self.path:
            return Image.open(self.path)
        return None


def swap_face(
    source_img: Image.Image,
    target_img: Image.Image,
    model: Union[str, None] = None,
    faces_index: Set[int] = {0},
    upscale_options: Union[UpscaleOptions, None] = None,
) -> ImageResult:
    result_image = target_img
    converted = convert_to_sd(target_img)
    scale, fn = converted[0], converted[1]
    if model is not None and not scale:
        if isinstance(source_img, str):  # source_img is a base64 string
            import base64, io
            if 'base64,' in source_img:  # check if the base64 string has a data URL scheme
                base64_data = source_img.split('base64,')[-1]
                img_bytes = base64.b64decode(base64_data)
            else:
                # if no data URL scheme, just decode
                img_bytes = base64.b64decode(source_img)
            source_img = Image.open(io.BytesIO(img_bytes))
        source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
        target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
        source_face = get_face_single(source_img, face_index=0)
        if source_face is not None:
            result = target_img
            model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), model)
            face_swapper = getFaceSwapModel(model_path)

            for face_num in faces_index:
                target_face = get_face_single(target_img, face_index=face_num)
                if target_face is not None:
                    result = face_swapper.get(result, target_face, source_face)
                else:
                    logger.info(f"No target face found for {face_num}")

            result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            if upscale_options is not None:
                result_image = upscale_image(result_image, upscale_options)
        else:
            logger.info("No source face found")
    result_image.save(fn.name)
    return ImageResult(path=fn.name)
