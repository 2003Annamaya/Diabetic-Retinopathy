"""
Diabetic Retinopathy – Inference / Prediction (PyTorch)
========================================================
Load a trained model and predict DR severity for new retinal images.
"""

import sys
import torch
import torch.nn.functional as F

import config as cfg
from models.retinopathy_model import build_model
from utils.preprocessing import prepare_single_image


def predict(image_path: str, model_path: str = cfg.BEST_MODEL_PATH):
    """Predict DR severity for a single retinal image."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    img = prepare_single_image(image_path).to(device)

    with torch.no_grad():
        outputs = model(img)
        probabilities = F.softmax(outputs, dim=1)[0].cpu().numpy()

    class_index = int(probabilities.argmax())
    class_name = cfg.CLASS_NAMES[class_index]
    confidence = float(probabilities[class_index])

    result = {
        "class_name": class_name,
        "class_index": class_index,
        "confidence": round(confidence * 100, 2),
        "all_probabilities": {
            name: round(float(prob) * 100, 2)
            for name, prob in zip(cfg.CLASS_NAMES, probabilities)
        },
    }
    return result


def severity_description(class_name: str) -> str:
    """Return a human-readable description of the DR severity level."""
    descriptions = {
        "No_DR": (
            "No signs of diabetic retinopathy detected. "
            "Continue regular eye check-ups as recommended by your doctor."
        ),
        "Mild": (
            "Mild non-proliferative diabetic retinopathy (NPDR). "
            "Small areas of balloon-like swelling (microaneurysms) in the retina's blood vessels. "
            "Follow-up with an ophthalmologist is recommended."
        ),
        "Moderate": (
            "Moderate NPDR. Blood vessels that nourish the retina may swell and distort. "
            "They may also lose their ability to transport blood. "
            "Consult an ophthalmologist soon."
        ),
        "Severe": (
            "Severe NPDR. Many blood vessels are blocked, depriving blood supply to the retina. "
            "These areas signal the retina to grow new blood vessels. "
            "Urgent ophthalmological attention required."
        ),
        "Proliferative": (
            "Proliferative diabetic retinopathy (PDR) – the most advanced stage. "
            "New, fragile blood vessels grow along the retina and may leak blood. "
            "Immediate medical intervention is critical."
        ),
    }
    return descriptions.get(class_name, "Unknown severity level.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_retinal_image>")
        sys.exit(1)

    image_file = sys.argv[1]
    result = predict(image_file)

    print("\n" + "=" * 50)
    print("  Diabetic Retinopathy Prediction Result")
    print("=" * 50)
    print(f"  Image      : {image_file}")
    print(f"  Prediction : {result['class_name']}")
    print(f"  Confidence : {result['confidence']}%")
    print("-" * 50)
    print("  Probabilities:")
    for cls, prob in result["all_probabilities"].items():
        bar = "█" * int(prob / 2)
        print(f"    {cls:<15} {prob:6.2f}%  {bar}")
    print("-" * 50)
    print(f"\n  ℹ  {severity_description(result['class_name'])}")
    print("=" * 50)
