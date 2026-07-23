# AI Synthetic Face Detector — Streamlit App

A demo UI for your MobileNetV2 fake/real face classifier (81.42% accuracy, AUC-ROC 0.891).

## What's in this folder
```
app/
├── app.py                  ← the Streamlit app
├── final_model.keras       ← your trained model (~26 MB)
├── requirements.txt
└── results/                ← plots shown on the "Model Performance" tab
```

## ⚠️ Before you deploy — check this one thing
The app assumes class index `0 = Fake`, `1 = Real` (alphabetical order, which is
what Keras' `ImageDataGenerator` / `image_dataset_from_directory` use by default).
If in your training notebook you printed `class_indices` and it showed something
different, open `app.py` and fix this line:
```python
CLASS_NAMES = ["Fake", "Real"]
```

## Run it locally first (2 minutes)
```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```
It'll open at `http://localhost:8501`. Upload a face photo and confirm the
prediction looks right before you deploy.

---

## Deploy for free — Streamlit Community Cloud (recommended)

This is the best option for a Keras/TensorFlow model — it's built for exactly
this, it's free, and you get one public link to share with your professor.
(Skip Vercel — it's built for JS apps and doesn't handle TensorFlow well.)

### Step 1 — Push this folder to GitHub
```bash
cd app
git init
git add .
git commit -m "Face detector Streamlit app"
```
Create a new **public** repo on GitHub (e.g. `face-detector-app`), then:
```bash
git remote add origin https://github.com/<your-username>/face-detector-app.git
git branch -M main
git push -u origin main
```
> Your model file is ~26 MB, well under GitHub's 100 MB limit, so a normal
> `git push` works fine — no Git LFS needed.

### Step 2 — Deploy on Streamlit Cloud
1. Go to **https://share.streamlit.io** and sign in with GitHub.
2. Click **"New app"**.
3. Pick your repo, branch `main`, and set the main file path to `app.py`.
4. Click **Deploy**.
5. Wait 2–5 minutes for the build (installing TensorFlow takes a little
   while) — then you'll get a public link like:
   `https://face-detector-app-<random>.streamlit.app`

That link is what you show your professor — one URL, works on any device,
no installation needed.

---

## Alternative: Hugging Face Spaces (also free, also great for ML)
1. Go to **https://huggingface.co/new-space**.
2. Name it, choose **Streamlit** as the SDK, set visibility to Public.
3. Upload `app.py`, `requirements.txt`, `final_model.keras`, and the
   `results/` folder (drag-and-drop in the web UI, or `git push` the same
   way as GitHub — Spaces are Git repos too).
4. It builds automatically and gives you a link like
   `https://huggingface.co/spaces/<you>/face-detector`.

This is a good backup if Streamlit Cloud's build queue is slow before your
presentation.

---

## Tips for the presentation
- Open the **"🔍 Try the Model"** tab and have 2–3 test images ready
  (one obviously real, one AI-generated) to demo live.
- Switch to **"📊 Model Performance"** to walk through the confusion
  matrix, ROC curve, and Grad-CAM visualizations — this shows the
  professor you understand *why* the model works, not just that it does.
- Mention the two-phase transfer-learning setup (`best_phase1.keras` →
  `best_phase2.keras` → `final_model.keras`) if asked about training strategy.
