# Signify-ISL 📚🧏‍♂️

**Signify-ISL** is a powerful, AI-driven communication and educational platform that translates Indian Sign Language (ISL) into text and speech in real-time. It's designed to support inclusive online education for the Deaf and hard-of-hearing community. Our mission is to make learning as seamless and accessible as possible for everyone—especially in virtual classroom environments.

🌐 [Live Demo](https://signify-isl.vercel.app/)

---

## 🔥 Features

- **🧠 Real-Time ISL Translation**  
  Uses computer vision and machine learning to detect Indian Sign Language and convert it into both text and speech instantly.

- **🎥 ISL-Powered Video Calling (Online Classroom)**  
  Host or join online video classrooms. Teachers can deliver lectures while ISL is translated on-screen, allowing students with hearing impairments to follow along in real-time.

- **🎓 Interactive ISL Learning Platform**  
  Learn Indian Sign Language through structured lessons, gesture demos, and quizzes. Track your learning progress.

- **🌍 Multi-Language Support**  
  Translate ISL to multiple Indian languages for broader inclusivity and understanding.

- **🔐 User Authentication**  
  Students and teachers can register and log in to personalize their learning or teaching experience.

  ### 🔄 Fallback Mechanism – Robustness Beyond AI

To ensure consistent and reliable performance, even in challenging real-world conditions (e.g., poor lighting, hand occlusion, or noisy backgrounds), **Signify ISL** integrates a fallback mechanism using **hand landmark tracking**.

If the primary **CNN model** is unable to classify a gesture confidently, the system automatically switches to an alternative logic:

- ✋ **Hand keypoints are captured using [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands)**
- 📐 The system analyzes **relative positions** of key landmarks — such as:
  - Fingertip distances  
  - Finger angles  
  - Hand orientation
- 🤖 Based on these geometric features, the system **infers the most likely ISL sign**

This fallback logic ensures that users continue to receive **accurate text and speech translations**, even under suboptimal conditions — increasing both the **reliability** and **usability** of the application.


---

## 🏫 Use Case: Online Classrooms for Deaf Students

This project can be extended into a **complete online learning platform like Physics Wallah**, where:

- Teachers conduct **live classes** via the video call interface.
- Students view **real-time ISL translations**.
- Additional features like **class scheduling, notes, assignments**, and **recorded lectures** can be integrated.

Ideal for schools, colleges, and online educators focused on **inclusive education**.

---

## 🛠️ Tech Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend & AI/ML**: Python, Flask, PyTorch , OpenCV
- **Hosting**: Vercel

---

## 🚀 How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/finefigo/signify-isl.git
   cd signify-isl

