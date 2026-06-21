# 🧠 NeuroFusionAI

> **Medical-Grade Brain Tumor Detection & Radiologist Assistance Platform**  
> Powered by Deep Learning · Built with Django · Ready for Production

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Django](https://img.shields.io/badge/Django-5.x-092E20?style=for-the-badge&logo=django&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-CPU-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5-7952B3?style=for-the-badge&logo=bootstrap&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Dev-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Production-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)

</div>

---

## What is NeuroFusionAI?

**NeuroFusionAI** is a full-stack, production-ready web application for **AI-powered brain tumor detection** from MRI scans. It is designed to assist radiologists, doctors, researchers, and the general public by providing:

- Instant tumor classification from MRI uploads
- Visual Grad-CAM heatmap overlays showing **where** the AI is looking
- Confidence scores and clinical risk levels
- Patient management, report generation, and analytics

This is not a simple demo — it is architected as a **real clinical tool** with role-based access, PDF reporting, audit trails, and deployment-ready infrastructure.

---

## What Makes NeuroFusionAI Different?

| Feature | NeuroFusionAI | Typical GitHub Projects |
|---|---|---|
| **Grad-CAM Heatmaps** | Real jet-colormap overlay on MRI | Just shows a label |
| **6 Tumor Classes** | Glioma, Meningioma, Pituitary, Schwannoma, Neurocytoma, Normal | Usually only 4 classes |
| **Sequential Model Grad-CAM** | Works with Sequential Keras models (custom submodel split) | Breaks on Sequential models |
| **Role-Based Access** | Admin / Radiologist / Doctor / Researcher / Public | Single user |
| **PDF Report Generation** | Auto-generated clinical PDF via ReportLab | Not available |
| **Patient Management** | Name, ID, Age, Gender, Notes per scan | Just file upload |
| **Analytics Dashboard** | Chart.js risk charts, scan trends, class distribution | None |
| **Google Sign-In** | OAuth2 via google-auth | Not available |
| **Docker Support** | Dockerfile + docker-compose.yml included | Not available |
| **Production-Ready** | Gunicorn + WhiteNoise + PostgreSQL + Cloudinary | Dev-only |
| **Email Integration** | Console / SMTP backends wired | Not available |
| **Internationalization** | Multi-language support built-in | English only |
| **Fallback Predictor** | Mock predictor if model fails to load | Crashes |
| **Audit Trail** | Timestamp + user linked to every scan | Not tracked |

---

## Tumor Classes Detected

| Class | Risk Level | Description |
|---|---|---|
| Normal | Low | No tumor detected |
| pituitary_tumor | Medium | Affects hormone regulation |
| Schwannoma_tumor | Medium | Nerve sheath tumor |
| glioma_tumor | High | Most aggressive brain tumor |
| meningioma_tumor | High | Arises from brain membranes |
| Neurocitoma_tumor | High | Rare intraventricular tumor |

---

## Project Structure

```
BrainTumorProject/
├── manage.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── Procfile                        # Heroku/Render deployment
├── runtime.txt                     # Python version pin
│
├── Project/                        # Django project config
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
│
├── App/                            # Main application
│   ├── models.py                   # Profile, ScanRecord
│   ├── forms.py                    # Registration, Upload, Profile forms
│   ├── views.py                    # All views + CNN prediction logic
│   ├── urls.py                     # URL routing
│   ├── admin.py                    # Django admin customization
│   ├── signals.py                  # Auto-create Profile on user creation
│   ├── context_processors.py       # Multi-language support
│   ├── translation_data.py         # Language strings
│   ├── tests.py                    # Unit tests
│   │
│   ├── Model/
│   │   └── brain_tumor_model.h5    # Trained CNN (6-class, 224x224 input)
│   │
│   ├── Dataset/
│   │   ├── TRAIN/                  # Training MRI images (6 classes)
│   │   └── TEST/                   # Test MRI images (6 classes)
│   │
│   ├── static/
│   │   ├── css/style.css           # Dark medical glassmorphism UI
│   │   ├── js/main.js
│   │   └── images/
│   │
│   └── templates/
│       ├── base.html               # Shared layout, navbar, sidebar
│       ├── app/
│       │   ├── dashboard.html      # Stats overview
│       │   ├── model.html          # MRI upload form
│       │   ├── output.html         # Prediction result + heatmap
│       │   ├── database.html       # Patient scan history
│       │   └── analytics.html      # Charts and trends
│       └── users/
│           ├── home.html           # Landing page
│           ├── login.html
│           ├── profile.html
│           └── change_password.html
│
└── media/
    ├── avatars/                    # User profile pictures
    └── scans/                      # Uploaded MRIs + Grad-CAM heatmaps
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.11, Django 5.x |
| **AI / ML** | TensorFlow CPU 2.x, Keras, NumPy |
| **Visualization** | Grad-CAM (custom Sequential-compatible), Matplotlib, Pillow |
| **Frontend** | Bootstrap 5, Chart.js, Vanilla JS, CSS Glassmorphism |
| **Database** | SQLite (dev), PostgreSQL (prod) |
| **Auth** | Django Auth + Google OAuth2 (google-auth) |
| **Storage** | Local media / Cloudinary (production) |
| **PDF** | ReportLab |
| **Server** | Gunicorn + WhiteNoise |
| **Container** | Docker + docker-compose |
| **Deploy** | Heroku / Render / Railway compatible |

---

## Local Setup

### Prerequisites
- Python 3.11
- pip
- Git

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/NeuroFusionAI.git
cd NeuroFusionAI
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Apply database migrations
```bash
python manage.py migrate
```

### 5. Create a superuser (admin)
```bash
python manage.py createsuperuser
```

### 6. Run the development server
```bash
python manage.py runserver
```

Visit: http://127.0.0.1:8000

---

## Docker Setup

```bash
# Build and start
docker-compose up --build

# Run migrations inside container
docker-compose exec web python manage.py migrate
docker-compose exec web python manage.py createsuperuser
```

---

## Production Deployment (Render / Railway / Heroku)

### Environment Variables to Set

| Variable | Value |
|---|---|
| SECRET_KEY | Your Django secret key |
| DEBUG | False |
| DATABASE_URL | PostgreSQL connection string |
| CLOUDINARY_URL | Cloudinary media storage URL |
| EMAIL_HOST_USER | SMTP email address |
| EMAIL_HOST_PASSWORD | SMTP app password |
| ALLOWED_HOSTS | yourdomain.com |

### Deploy Steps
```bash
# Collect static files
python manage.py collectstatic --noinput

# Start with Gunicorn
gunicorn Project.wsgi:application
```

---

## Running Tests

```bash
python manage.py test App
```

Tests include:
- User registration flow
- Login / logout
- Authenticated upload access control
- Prediction fallback accuracy range
- API endpoint responses

---

## AI Model Details

### Architecture
- **Type:** Sequential CNN (Convolutional Neural Network)
- **Input:** (224, 224, 3) — RGB MRI image
- **Output:** (6,) — softmax probabilities across 6 classes
- **Layers:** conv2d_1 > MaxPool > conv2d_2 > MaxPool > conv2d_final > GlobalAvgPool > Dense > predictions

### Grad-CAM Visualization
NeuroFusionAI uses a **custom Sequential-compatible Grad-CAM implementation** that:
1. Splits the model into two sub-models around conv2d_final
2. Uses tf.GradientTape to track gradients w.r.t. conv feature maps
3. Pools gradients and creates a jet-colormap heatmap overlay
4. Blends the heatmap with the original MRI at 40% opacity

Most open-source implementations break on Sequential models. NeuroFusionAI solves this with a submodel-split approach.

### Dataset
```
App/Dataset/TRAIN/<class_name>/  → Training images
App/Dataset/TEST/<class_name>/   → Test images
```
Classes: glioma_tumor, meningioma_tumor, Neurocitoma_tumor, Normal, pituitary_tumor, Schwannoma_tumor

---

## User Roles

| Role | Permissions |
|---|---|
| **Admin** | Full access, all scans, all users |
| **Radiologist** | Upload scans, view own history, add clinical notes |
| **Doctor** | Upload scans, view own patients |
| **Researcher** | View all scan history and analytics |
| **Public** | Upload and view own results only |

---

## Key Features

### MRI Upload and Prediction
- Upload JPEG/PNG MRI images
- Instant AI classification (6 tumor types)
- Confidence score (0-100%)
- Risk level badge: Low / Medium / High
- Grad-CAM heatmap overlay highlighting tumor region

### Patient Management
- Patient name, ID, age, gender
- Clinical notes by radiologist
- Full scan history with search and filter

### Analytics Dashboard
- Total scans, risk distribution (Chart.js doughnut)
- Scan upload trend (line chart)
- Tumor class distribution (bar chart)

### PDF Report
- Auto-generated clinical report per scan
- Includes patient info, prediction result, risk level, timestamp

---

## Security Features

- CSRF protection on all forms
- login_required on all sensitive views
- Role-based data isolation (users see only their own scans unless admin/researcher)
- DEBUG = False safe for production
- WhiteNoise for secure static file serving
- SECRET_KEY managed via environment variable in production

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add: your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## License

This project is licensed under the **MIT License**.

---

## Author

**Dhevika**
Built with love using Django, TensorFlow, and a strong belief that AI can assist medicine — not replace doctors.

---

NeuroFusionAI — Where Neuroscience meets Artificial Intelligence
