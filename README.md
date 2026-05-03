# VisionML Studio: Browser-Based Computer Vision Platform

![Version](https://img.shields.io/badge/version-v1.0.0-blue)
![Status](https://img.shields.io/badge/status-active-success)

VisionML Studio is an interactive, browser-based machine learning application that empowers users to explore computer vision without writing complex code. Users can instantly classify objects using a pre-trained MobileNet model or build, train, and test their own custom Convolutional Neural Network (CNN) using transfer learning directly in the browser.

**Developed by Nitin Kumar Saini**

---

## 📸 Previews

### Quick Predict (MobileNet)
![Quick Predict](./src/VisionML%20Studio%20-%20Quick%20Predict%20%28Pre-trained%20MobileNet%29.png)

### Custom CNN Training Dashboard
![Custom Training](./src/VisionML%20Studio%20-%20Training%20Dashboard.png)

### Test Your Custom Model
![Test Custom Model](./src/VisionML%20Studio%20-%20Test%20Your%20Custom%20Model.png)

### Classification Results
![Results](./src/VisionML%20Studio%20Prediction.png)

---

## ✨ Features

*   **Quick Predict (Pre-trained):** Instantly classify general objects using a pre-trained MobileNet model built on the 1000 original ImageNet classes.
*   **Custom Transfer Learning:** Define your own categories, build a dataset using your webcam or image uploads, and train a custom CNN model on the fly.
*   **Live Training Dashboard:** Monitor your custom model's learning progress with real-time visualization of Accuracy and Loss metrics.
*   **Flexible Inputs:** Supports both drag-and-drop file uploads (JPG, PNG) and live web-camera feeds for dynamic image classification.
*   **Interactive Testing:** Seamlessly move from training a custom model to testing it with new, unseen data in a unified workflow.

## 🛠️ Technologies Used

*   **Frontend Framework:** React with Vite
*   **Machine Learning:** TensorFlow.js (MobileNet & Transfer Learning)
*   **Language:** TypeScript / JavaScript
*   **Styling:** CSS / Tailwind CSS

## 🚀 Getting Started

Follow these instructions to set up the project locally on your machine.

### Prerequisites

*   Node.js installed on your machine

### Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nitinsaini1805/visionml-studio.git
   cd visionml-studio
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm run dev
   ```

## 👨‍💻 Author

**Nitin Kumar Saini**
*   GitHub: [@nitinsaini1805](https://github.com/nitinsaini1805)

## 📝 License

This project is for educational and portfolio purposes.

Once you save the file, open your terminal in VS Code and run these three commands to push the perfect, final version to your GitHub:

```bash
git add .
git commit -m "Updated README with exact screenshot file paths"
git push
```