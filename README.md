
InteriorAI - Room Type Classifier

This project implements a robust image classification system designed to categorize indoor room images (Bedroom, Kitchen, Living Room, etc.) with high precision. It utilizes Transfer Learning with a ResNet50 model pre-trained on ImageNet.

The pipeline is optimized for efficiency, making it suitable for deployment on edge devices via TensorFlow Lite or Python backends.

🚀 Key Features

ResNet50 Backbone: Leverages deep feature extraction from ImageNet weights.

CPU Optimized: Tuned batch sizes and worker threads for standard hardware.

Inference Ready: Includes a standalone script (test_room_classifier.py) to classify new images immediately.

📊 Results

Accuracy: Achieved ~80% accuracy on the validation set after fine-tuning.

Classes: Capable of distinguishing between Bathroom, Bedroom, Dining Room, Kitchen, and Living Room.

🛠️ Tech Stack

Python 3.10+ | PyTorch | NumPy (<2.0) | Matplotlib | Scikit-learn

🤝 Contributing

Contributions are welcome via Pull Request.
