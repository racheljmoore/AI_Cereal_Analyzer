import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace  # type: ignore

def analyze_cereal_taster(image_path):
    try:
        # Load the image
        img = cv2.imread(image_path)  # Read image
        if img is None:
            raise ValueError("Image not found. Check the file path.")

        # Display the image
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

        # Perform DeepFace analysis
        result = DeepFace.analyze(image_path, actions=['age', 'emotion'])

        # Extract age and dominant emotion
        age = result[0]['age']
        emotion = result[0]['dominant_emotion']

        print(f"Detected Age: {age}")
        print(f"Detected Emotion: {emotion}")

        return age, emotion

    except Exception as e:
        print("Error:", e)

# Test the function with an image
if __name__ == "__main__":
    image_file = "C:/Users/Rachel/Desktop/Python/AI_Cereal_Analyzer/Sample_test_Face.jpg"
    analyze_cereal_taster(image_file)

