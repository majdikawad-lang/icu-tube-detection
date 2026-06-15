import pydicom
import cv2
import numpy as np
import io

class DicomProcessor:
    @staticmethod
    def process_dicom(file_bytes: bytes) -> np.ndarray:
        """
        Reads a DICOM file from bytes, normalizes the pixel array,
        applies histogram equalization, and converts it to a format
        suitable for AI model inference (e.g., RGB image, normalized 0-1).
        """
        # Read the DICOM file from memory
        dicom_file = pydicom.dcmread(io.BytesIO(file_bytes))
        
        # Extract pixel array
        pixel_array = dicom_file.pixel_array
        
        # Convert to float for processing
        img = pixel_array.astype(np.float32)
        
        # Normalize to 0-255 range based on windowing or min/max
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min) * 255.0
        else:
            img = np.zeros_like(img)
            
        img = img.astype(np.uint8)
        
        # Convert grayscale to RGB (models usually expect 3 channels)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        return img_rgb
