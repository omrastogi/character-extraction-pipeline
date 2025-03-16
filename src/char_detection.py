import cv2
import os
from imgutils.detect import detect_person

class PersonCropper:
    def __init__(self):
        """Initialize the person detector and cropper."""
        pass

    def crop_persons(self, image_path, output_dir):
        """Detect and crop persons from an image, saving them to an output directory."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        image = cv2.imread(image_path)
        detections = detect_person(image_path)
        cropped_images = []
        
        for idx, (bbox, label, confidence) in enumerate(detections):
            if label == "person":
                x1, y1, x2, y2 = bbox
                cropped_image = image[y1:y2, x1:x2]
                output_path = os.path.join(output_dir, f"cropped_person_{idx}.jpg")
                cv2.imwrite(output_path, cropped_image)
                cropped_images.append(output_path)
        
        return cropped_images

# Example Usage
if __name__ == "__main__":
    cropper = PersonCropper()
    image_path = "/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/character-extraction-pipeline/data/continued/sensitive/danbooru_1370513_e8f30add09fdad6eb332b284f4a408bd.jpg"
    output_dir = "cropped_persons"
    
    cropped_faces = cropper.crop_persons(image_path, output_dir)
    if cropped_faces:
        print(f"Cropped {len(cropped_faces)} persons. Saved to {output_dir}.")
    else:
        print("No valid persons detected.")


