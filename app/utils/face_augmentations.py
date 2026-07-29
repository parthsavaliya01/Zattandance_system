# app/utils/face_augmentations.py
import cv2
import numpy as np

def generate_face_variants(face_img: np.ndarray) -> list:
    """
    Generates safely augmented variants of the face for enrollment.
    Includes Original, Dark Gamma, and Bright Gamma.
    """
    variants = [
        face_img  # original
    ]

    # 1. Darker Variant (Gamma Correction)
    gamma_dark = 2.0
    invGamma_dark = 1.0 / gamma_dark
    table_dark = np.array([((i / 255.0) ** invGamma_dark) * 255 for i in np.arange(0, 256)]).astype("uint8")
    dark_img = cv2.LUT(face_img, table_dark)
    variants.append(dark_img)

    # 2. Brighter Variant (Gamma Correction)
    gamma_bright = 0.5
    invGamma_bright = 1.0 / gamma_bright
    table_bright = np.array([((i / 255.0) ** invGamma_bright) * 255 for i in np.arange(0, 256)]).astype("uint8")
    bright_img = cv2.LUT(face_img, table_bright)
    variants.append(bright_img)

    # Save variants to a folder so the user can see them
    import os
    import uuid
    
    save_dir = "outputs/enrollment_variants"
    os.makedirs(save_dir, exist_ok=True)
    
    # Use a unique ID so bulk enrollments don't overwrite each other
    uid = uuid.uuid4().hex[:6]
    
    # cv2.imwrite expects BGR, but variants are RGB. Convert before saving!
    cv2.imwrite(os.path.join(save_dir, f"{uid}_original.jpg"), cv2.cvtColor(variants[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(save_dir, f"{uid}_dark.jpg"), cv2.cvtColor(variants[1], cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(save_dir, f"{uid}_bright.jpg"), cv2.cvtColor(variants[2], cv2.COLOR_RGB2BGR))

    return variants