
# 🌱 AgriScan AI – Plant Disease Detection Web App

AgriScan AI is a user-friendly web application that empowers farmers to detect plant diseases by simply uploading a photo of a leaf. The app uses a TensorFlow Lite (TFLite) deep learning model to predict diseases in real-time and provides AI-assisted answers through a built-in ChatGPT-powered assistant.

 Key Features

 **Leaf Scan Diagnosis** – Upload a photo of a leaf to detect plant diseases.
 **AI Chatbot Assistant** – Ask agriculture-related questions and get intelligent responses.
   **User Authentication** – Secure login/signup system.
   **Web Interface** – Clean, mobile-friendly UI with sidebar navigation and footer.
   **TFLite Model Integration** – Lightweight model for fast predictions.
   **Live Chat** – Integrated with OpenAI for smart Q&A.

---

 Live Demo

🖥 [https://agriscan-z636.onrender.com] *(example URL, replace with your deployed app)*
https://youtu.be/Yf9EGg_BpWc (YOUTUBE LINK)
https://youtu.be/odj_bJTLFiU  (tap this link for a live demo)

---


 Tech Stack

| Component       | Tech Used                            |
|-----------------|---------------------------------------|
| Frontend        | HTML, CSS, JavaScript                 |
| Backend         | Python (Flask)                        |
| AI Model        | TensorFlow Lite (plant disease model) |
| Auth            | SQLite, SQLAlchemy, Passlib           |
| Chatbot         | OpenAI (GPT-3.5 or GPT-4)             |
| Deployment      | Render.com                            |
| Image Handling  | Pillow, NumPy                         |

 Project Structure

├── app.py # Main Flask application
├── plant_disease_model.tflite
├── label_map.txt # Class names
├── agriscan_users.db # SQLite DB (generated)
├── requirements.txt
├── .env # API keys (not uploaded)
├── templates/ # HTML templates (if separated)
├── static/ # CSS, JS, images (if separated)
├── screenshots/ # Optional: for README

---

 Model

- Trained TensorFlow model exported to `.tflite` for lightweight inference.
- Input size is dynamically set from model metadata.
- Model returns:
  - `class_`: Disease name or healthy status.
  - `confidence`: Prediction confidence percentage.

---

 Environment Variables

Create a `.env` file and add the following:

```env
SECRET_KEY=your_flask_secret_key
OPENAI_API_KEY=your_openai_api_key
 Installation & Local Setup
 1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/yourusername/agriscan-ai.git
cd agriscan-ai
 2. Create a Virtual Environment
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
 3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
 4. Add .env File
Add your OPENAI_API_KEY and SECRET_KEY as shown above.

 5. Run the App
bash
Copy
Edit
python app.py
Visit http://127.0.0.1:5000 in your browser.


 Future Improvements
 Offline/PWA version for farmers with no internet.

 Translate to local languages like Kiswahili.

 Add more crop types and retrain with custom datasets.

 Integrate with satellite weather data (NDVI, rainfall, etc).

 Build Android app using Kivy or Flutter.

 Contributing
Pull requests are welcome! If you're interested in expanding the dataset, improving the UI, or integrating new features, feel free to fork and PR.
 License
MIT License © 2025 [Albert Spoi]

 Contact
Email: albsipoi1564@gmail.com

LinkedIn: 

GitHub: github.com/Dr-Alb

Built with ❤️ for smallholder farmers and the future of sustainable agriculture.
