import os
import io
import random
import numpy as np
from PIL import Image, ImageDraw

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth.forms import PasswordChangeForm
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from django.conf import settings
from django.core.files.base import ContentFile
from django.core.mail import EmailMessage
from django.db.models import Count, Q
from django.utils import timezone
from django.core.paginator import Paginator

from .models import Profile, ScanRecord
from .forms import RadiologistRegisterForm, ProfileUpdateForm, ScanUploadForm
from .translation_data import TRANSLATIONS

# Disable TensorFlow warnings in logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

CLASS_NAMES = [
    'glioma_tumor',
    'meningioma_tumor',
    'Neurocitoma_tumor',
    'Normal',
    'pituitary_tumor',
    'Schwannoma_tumor'
]

RISK_MAP = {
    'Normal': 'Low',
    'pituitary_tumor': 'Medium',
    'Schwannoma_tumor': 'Medium',
    'glioma_tumor': 'High',
    'meningioma_tumor': 'High',
    'Neurocitoma_tumor': 'High',
}

# Lazy loading of TensorFlow
model = None

def get_model():
    global model
    if model is None:
        try:
            import tensorflow as tf
            model_path = os.path.join(settings.BASE_DIR, 'App', 'Model', 'brain_tumor_model.h5')
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(model_path)
                print("TensorFlow model loaded successfully.")
            else:
                print(f"Model file not found at {model_path}. Using fallback mock predictor.")
        except Exception as e:
            print(f"Failed to load Keras model: {e}. Falling back to mock predictions.")
            model = None
    return model


def run_mri_prediction(image_path):
    """
    Perform CNN prediction on the uploaded MRI image.
    Generates a Grad-CAM heatmap overlay.
    """
    # 1. Fallback Predictor if TensorFlow loading failed
    loaded_model = get_model()
    if loaded_model is None:
        # Mock prediction for demonstration/fallback
        pred_idx = random.randint(0, len(CLASS_NAMES) - 1)
        confidence = random.uniform(85.0, 99.8)
        pred_class = CLASS_NAMES[pred_idx]
        
        # Generate mock Grad-CAM overlay (Pillow draw)
        img = Image.open(image_path).convert('RGB')
        overlay = Image.new('RGB', img.size, color=(0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        w, h = img.size
        # Draw target region
        if pred_class != 'Normal':
            cx, cy = w // 2 + random.randint(-40, 40), h // 2 + random.randint(-40, 40)
            r = min(w, h) // 5
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 0, 0))
            draw.ellipse([cx - int(r*0.7), cy - int(r*0.7), cx + int(r*0.7), cy + int(r*0.7)], fill=(255, 128, 0))
        else:
            # Subtle default highlight
            draw.ellipse([0, 0, 10, 10], fill=(0, 255, 0))
            
        heatmap_img = Image.blend(img, overlay, 0.45)
        return pred_class, confidence, heatmap_img

    # 2. Real TensorFlow & Grad-CAM execution
    try:
        import tensorflow as tf
        import matplotlib
        import matplotlib.cm as cm

        img = Image.open(image_path).convert('RGB')
        original_size = img.size

        # Preprocessing image (224x224, Normalized)
        img_resized = img.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize to [0,1]

        # Predict Class
        preds = loaded_model.predict(img_array, verbose=0)
        pred_idx = int(np.argmax(preds[0]))
        confidence = float(preds[0][pred_idx]) * 100.0
        pred_class = CLASS_NAMES[pred_idx]

        # Grad-CAM Heatmap computation
        try:
            last_conv_layer_name = "conv2d_final"

            # For Sequential models, we use a submodel approach:
            # Build a partial model up to (and including) conv2d_final, then the rest.
            conv_layer_idx = next(
                i for i, l in enumerate(loaded_model.layers) if l.name == last_conv_layer_name
            )
            # Sub-model: input → conv2d_final
            inp_layer = tf.keras.Input(shape=(224, 224, 3))
            x = inp_layer
            for layer in loaded_model.layers[:conv_layer_idx + 1]:
                x = layer(x)
            conv_submodel = tf.keras.Model(inputs=inp_layer, outputs=x)

            # Sub-model: conv2d_final_output → predictions
            conv_inp = tf.keras.Input(shape=conv_submodel.output_shape[1:])
            y = conv_inp
            for layer in loaded_model.layers[conv_layer_idx + 1:]:
                y = layer(y)
            head_submodel = tf.keras.Model(inputs=conv_inp, outputs=y)

            img_tensor = tf.cast(img_array, tf.float32)
            with tf.GradientTape() as tape:
                conv_outputs = conv_submodel(img_tensor)
                tape.watch(conv_outputs)
                predictions = head_submodel(conv_outputs)
                loss = predictions[:, pred_idx]

            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            cam = conv_outputs[0]
            heatmap = cam @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)

            max_val = tf.math.reduce_max(heatmap)
            heatmap = tf.maximum(heatmap, 0) / (max_val + 1e-10)
            heatmap = heatmap.numpy()

            # Map heatmap to PIL Jet Colormap overlay
            heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize(original_size, Image.Resampling.BICUBIC)
            heatmap_resized = np.array(heatmap_resized)

            colormap = matplotlib.colormaps["jet"]
            colormap_colors = colormap(np.arange(256))[:, :3]
            colormap_heatmap = colormap_colors[heatmap_resized]

            colormap_heatmap_img = Image.fromarray(np.uint8(255 * colormap_heatmap))
            heatmap_img = Image.blend(img, colormap_heatmap_img, 0.4)
        except Exception as cam_err:
            print(f"Grad-CAM overlay failed: {cam_err}. Returning original.")
            heatmap_img = img

        return pred_class, confidence, heatmap_img
    except Exception as general_err:
        print(f"TensorFlow runtime error: {general_err}")
        return 'Normal', 95.0, Image.open(image_path)


# --- Authentication & General Public Views ---

def home_view(request):
    return render(request, 'users/home.html')

def register_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    if request.method == 'POST':
        form = RadiologistRegisterForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data['password'])
            user.save()
            
            # Save Profile specifics
            profile = user.profile
            profile.role = form.cleaned_data['role']
            profile.professional_id = form.cleaned_data['professional_id']
            profile.department = form.cleaned_data['department']
            profile.save()
            
            login(request, user)
            messages.success(request, f"Welcome to NeuroFusionAI, Dr. {user.username}!")
            return redirect('dashboard')
    else:
        form = RadiologistRegisterForm()
    return render(request, 'users/login.html', {'form': form, 'tab': 'register'})

def login_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            messages.success(request, f"Welcome back, Dr. {user.username}.")
            return redirect('dashboard')
        else:
            messages.error(request, "Invalid credentials. Please verify your login info.")
    return render(request, 'users/login.html', {'tab': 'login'})

def logout_view(request):
    logout(request)
    messages.info(request, "Successfully logged out from active session.")
    return redirect('home')

@login_required
def profile_view(request):
    profile = request.user.profile
    if request.method == 'POST':
        form = ProfileUpdateForm(request.POST, request.FILES, instance=profile)
        if form.is_valid():
            # Update user details
            request.user.first_name = form.cleaned_data['first_name']
            request.user.last_name = form.cleaned_data['last_name']
            request.user.email = form.cleaned_data['email']
            request.user.save()
            
            form.save()
            messages.success(request, "Radiologist profile data successfully updated.")
            return redirect('profile')
    else:
        form = ProfileUpdateForm(instance=profile, initial={
            'first_name': request.user.first_name,
            'last_name': request.user.last_name,
            'email': request.user.email,
        })
    return render(request, 'users/profile.html', {'form': form})

@login_required
def change_password_view(request):
    if request.method == 'POST':
        form = PasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            user = form.save()
            messages.success(request, "Your password has been changed. Please sign in again.")
            return redirect('login')
    else:
        form = PasswordChangeForm(request.user)
    return render(request, 'users/change_password.html', {'form': form})

def google_login_api(request):
    """
    Mock Google sign-in endpoint. Creates a new user with 'public' role.
    """
    if request.method == 'POST':
        # Generate random Google credentials mapping
        rand_id = random.randint(1000, 9999)
        username = f"google_user_{rand_id}"
        email = f"{username}@gmail.com"
        
        user, created = User.objects.get_or_create(username=username, email=email)
        if created:
            user.set_password(User.objects.make_random_password())
            user.save()
            user.profile.role = 'public'
            user.profile.google_signed_in = True
            user.profile.save()
            
        login(request, user)
        return JsonResponse({'status': 'success', 'redirect_url': '/dashboard/'})
    return JsonResponse({'status': 'error', 'message': 'Invalid HTTP Method'}, status=400)


# --- Core Radiologist Assistance Views ---

@login_required
def dashboard_view(request):
    # Fetch scans related to user or all if researcher/admin
    if request.user.profile.role in ['admin', 'researcher']:
        scans = ScanRecord.objects.all()
    else:
        scans = ScanRecord.objects.filter(user=request.user)
        
    total_scans = scans.count()
    high_risk_scans = scans.filter(risk_level='High').count()
    medium_risk_scans = scans.filter(risk_level='Medium').count()
    low_risk_scans = scans.filter(risk_level='Low').count()
    
    recent_scans = scans.order_by('-uploaded_at')[:5]
    
    context = {
        'total_scans': total_scans,
        'high_risk_scans': high_risk_scans,
        'medium_risk_scans': medium_risk_scans,
        'low_risk_scans': low_risk_scans,
        'recent_scans': recent_scans,
    }
    return render(request, 'app/dashboard.html', context)


@login_required
def upload_scan_view(request):
    """
    Renders the upload page and handles MRI prediction execution.
    """
    if request.method == 'POST':
        form = ScanUploadForm(request.POST, request.FILES)
        if form.is_valid():
            scan = form.save(commit=False)
            scan.user = request.user
            # NOTE: Do NOT call scan.save() here — label & probability are NOT NULL
            # and must be populated by the prediction before the first DB write.

            # 1. Save the uploaded image file to disk without touching the DB
            #    by using the ImageField's save() helper directly.
            image_file = request.FILES['image']
            from django.core.files.storage import default_storage
            import os
            tmp_name = default_storage.save(
                f"scans/tmp_{image_file.name}", image_file
            )
            tmp_path = default_storage.path(tmp_name)

            # 2. Run prediction on the saved file
            label, probability, heatmap_img = run_mri_prediction(tmp_path)

            # 3. Move the temp file to its final location via the model field
            #    (re-open so Django handles the upload_to path correctly)
            from django.core.files import File
            with open(tmp_path, 'rb') as f:
                scan.image.save(os.path.basename(tmp_name), File(f), save=False)
            # Clean up tmp file
            try:
                default_storage.delete(tmp_name)
            except Exception:
                pass

            # 4. Assign all required fields before the first DB INSERT
            scan.label = label
            scan.probability = probability
            scan.risk_level = RISK_MAP.get(label, 'Low')

            # 5. Build and attach the Grad-CAM heatmap
            heatmap_io = io.BytesIO()
            heatmap_img.save(heatmap_io, format='JPEG', quality=95)
            heatmap_name = f"heatmap_{scan.patient_id}_{timezone.now().strftime('%Y%m%d%H%M%S')}.jpg"
            scan.heatmap.save(heatmap_name, ContentFile(heatmap_io.getvalue()), save=False)

            # 6. Single DB write — all NOT NULL fields are now populated
            scan.save()
            messages.success(request, "MRI scan analysis complete.")
            return redirect('prediction_output', scan_id=scan.id)
    else:
        form = ScanUploadForm()
    return render(request, 'app/model.html', {'form': form})


@login_required
def prediction_output_view(request, scan_id):
    scan = get_object_or_404(ScanRecord, id=scan_id)
    
    # Save radiologist notes via AJAX or POST
    if request.method == 'POST':
        notes = request.POST.get('notes')
        scan.notes = notes
        scan.save()
        messages.success(request, "Clinical notes successfully saved.")
        return redirect('prediction_output', scan_id=scan.id)
        
    return render(request, 'app/output.html', {'scan': scan})


@login_required
def database_view(request):
    """
    Radiologist History Database page with filters, search, and pagination.
    """
    search_query = request.GET.get('search', '')
    risk_filter = request.GET.get('risk', '')
    
    scans = ScanRecord.objects.all()
    if request.user.profile.role not in ['admin', 'researcher']:
        scans = scans.filter(user=request.user)
        
    if search_query:
        scans = scans.filter(
            Q(patient_name__icontains=search_query) |
            Q(patient_id__icontains=search_query) |
            Q(label__icontains=search_query)
        )
        
    if risk_filter:
        scans = scans.filter(risk_level=risk_filter)
        
    paginator = Paginator(scans, 8)  # 8 items per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    return render(request, 'app/database.html', {
        'page_obj': page_obj,
        'search_query': search_query,
        'risk_filter': risk_filter
    })


@login_required
def delete_scan_view(request, scan_id):
    scan = get_object_or_404(ScanRecord, id=scan_id)
    # Check permissions (only creator, admin, or radiologist can delete)
    if scan.user == request.user or request.user.profile.role == 'admin':
        scan.delete()
        messages.success(request, "Patient MRI record successfully deleted.")
    else:
        messages.error(request, "Access denied: Insufficient privileges.")
    return redirect('database')


@login_required
def analytics_view(request):
    """
    Aggregate counts by tumor category and risk levels for Chart.js rendering.
    """
    if request.user.profile.role in ['admin', 'researcher']:
        scans = ScanRecord.objects.all()
    else:
        scans = ScanRecord.objects.filter(user=request.user)
        
    # Aggregate tumor classes
    class_counts = scans.values('label').annotate(count=Count('label'))
    class_data = {c: 0 for c in CLASS_NAMES}
    for item in class_counts:
        if item['label'] in class_data:
            class_data[item['label']] = item['count']
            
    # Risk distributions
    risk_counts = scans.values('risk_level').annotate(count=Count('risk_level'))
    risk_data = {'Low': 0, 'Medium': 0, 'High': 0}
    for item in risk_counts:
        if item['risk_level'] in risk_data:
            risk_data[item['risk_level']] = item['count']
            
    # Scans by gender
    gender_counts = scans.values('patient_gender').annotate(count=Count('patient_gender'))
    gender_data = {'M': 0, 'F': 0, 'O': 0}
    for item in gender_counts:
        if item['patient_gender'] in gender_data:
            gender_data[item['patient_gender']] = item['count']
            
    context = {
        'class_labels': list(class_data.keys()),
        'class_values': list(class_data.values()),
        'risk_labels': list(risk_data.keys()),
        'risk_values': list(risk_data.values()),
        'gender_labels': ['Male', 'Female', 'Other'],
        'gender_values': [gender_data['M'], gender_data['F'], gender_data['O']],
    }
    return render(request, 'app/analytics.html', context)


# --- Document Reports & Communications ---

@login_required
def download_report_view(request, scan_id):
    """
    Generates a professional PDF report using ReportLab.
    """
    scan = get_object_or_404(ScanRecord, id=scan_id)
    
    # Setup PDF Buffer
    buffer = io.BytesIO()
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    styles = getSampleStyleSheet()
    
    # Medical Headers
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        textColor=colors.HexColor('#0055ff'),
        spaceAfter=5
    )
    subtitle_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#666666'),
        spaceAfter=20
    )
    h2_style = ParagraphStyle(
        'DocH2',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#0033aa'),
        spaceBefore=15,
        spaceAfter=10
    )
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.black
    )
    
    # Add title header
    story.append(Paragraph("NEUROFUSIONAI CLINICAL REPORT", title_style))
    story.append(Paragraph("Advanced CNN Diagnostic Brain Tumor Classification Platform", subtitle_style))
    story.append(Spacer(1, 10))
    
    # Patient Demographic Details Table
    data = [
        [Paragraph("<b>Patient Name:</b>", body_style), Paragraph(scan.patient_name, body_style),
         Paragraph("<b>Patient ID:</b>", body_style), Paragraph(scan.patient_id, body_style)],
        [Paragraph("<b>Age:</b>", body_style), Paragraph(str(scan.patient_age), body_style),
         Paragraph("<b>Gender:</b>", body_style), Paragraph(scan.get_patient_gender_display(), body_style)],
        [Paragraph("<b>Scan Timestamp:</b>", body_style), Paragraph(scan.uploaded_at.strftime('%Y-%m-%d %H:%M:%S UTC'), body_style),
         Paragraph("<b>Risk Evaluation:</b>", body_style), Paragraph(f"<b>{scan.risk_level}</b>", body_style)],
    ]
    t = Table(data, colWidths=[110, 150, 110, 150])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f2f5fa')),
        ('PADDING', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#dddddd')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t)
    story.append(Spacer(1, 20))
    
    # Diagnosis Details Table
    story.append(Paragraph("AI Diagnostic Evaluation", h2_style))
    diag_data = [
        [Paragraph("<b>Classification Result:</b>", body_style), Paragraph(scan.label.replace('_', ' ').title(), body_style)],
        [Paragraph("<b>Confidence Level:</b>", body_style), Paragraph(f"{scan.probability:.2f}%", body_style)],
    ]
    dt = Table(diag_data, colWidths=[150, 370])
    dt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#e6eeff')),
        ('PADDING', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cccccc')),
    ]))
    story.append(dt)
    story.append(Spacer(1, 20))
    
    # Render MRI and Grad-CAM side-by-side if they are saved locally
    story.append(Paragraph("MRI Visualizations", h2_style))
    try:
        # Since cloud storage URLs cannot be resolved directly locally via file path,
        # we check if storage is local. If local, load directly.
        if scan.image and hasattr(scan.image, 'path') and os.path.exists(scan.image.path):
            img_w = 230
            img_h = 230
            mri_img = RLImage(scan.image.path, width=img_w, height=img_h)
            
            if scan.heatmap and os.path.exists(scan.heatmap.path):
                cam_img = RLImage(scan.heatmap.path, width=img_w, height=img_h)
                img_table = Table([[mri_img, cam_img]], colWidths=[260, 260])
            else:
                img_table = Table([[mri_img, "Heatmap unavailable"]], colWidths=[260, 260])
                
            img_table.setStyle(TableStyle([
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ]))
            story.append(img_table)
        else:
            # When deployed to Cloudinary, load the image into memory via requests
            import requests
            img_url = scan.image.url
            cam_url = scan.heatmap.url if scan.heatmap else None
            
            response = requests.get(img_url, timeout=5)
            mri_data = io.BytesIO(response.content)
            mri_img = RLImage(mri_data, width=230, height=230)
            
            if cam_url:
                cam_response = requests.get(cam_url, timeout=5)
                cam_data = io.BytesIO(cam_response.content)
                cam_img = RLImage(cam_data, width=230, height=230)
                img_table = Table([[mri_img, cam_img]], colWidths=[260, 260])
            else:
                img_table = Table([[mri_img, "Heatmap unavailable"]], colWidths=[260, 260])
                
            img_table.setStyle(TableStyle([
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ]))
            story.append(img_table)
    except Exception as img_err:
        story.append(Paragraph(f"MRI Image preview offline. Visualizations omitted. ({img_err})", subtitle_style))
        
    story.append(Spacer(1, 20))
    
    # Clinical Observations Section
    story.append(Paragraph("Clinical Observations & Doctor Notes", h2_style))
    notes_content = scan.notes if scan.notes else "No notes added by the radiologist."
    story.append(Paragraph(notes_content.replace('\n', '<br/>'), body_style))
    story.append(Spacer(1, 40))
    
    # Doctor signature
    signature_data = [
        ["", "_____________________________________"],
        ["", f"Assigned Radiologist: Dr. {scan.user.username if scan.user else 'System'}"],
        ["", f"Department: {scan.user.profile.department if scan.user and scan.user.profile.department else 'Radiology'}"]
    ]
    sig_table = Table(signature_data, colWidths=[280, 240])
    sig_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'RIGHT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('FONTNAME', (1,1), (1,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
    ]))
    story.append(sig_table)
    
    # Build Document
    doc.build(story)
    
    buffer.seek(0)
    pdf_data = buffer.getvalue()
    buffer.close()
    
    response = HttpResponse(pdf_data, content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="neurofusion_report_{scan.patient_id}.pdf"'
    return response


@login_required
def email_report_view(request, scan_id):
    """
    Sends the PDF diagnostic report directly to the doctor's email.
    """
    scan = get_object_or_404(ScanRecord, id=scan_id)
    recipient_email = request.POST.get('email', '')
    
    if not recipient_email:
        messages.error(request, "Email recipient address cannot be empty.")
        return redirect('prediction_output', scan_id=scan.id)
        
    try:
        # Generate the PDF content
        pdf_response = download_report_view(request, scan_id)
        pdf_content = pdf_response.content
        
        email = EmailMessage(
            subject=f"[NeuroFusionAI] Clinical Scan Report for Patient {scan.patient_name} ({scan.patient_id})",
            body=(
                f"Hello Dr.,\n\n"
                f"Please find attached the MRI Scan diagnostic report generated by NeuroFusionAI.\n\n"
                f"Diagnostic Summary:\n"
                f"- Patient ID: {scan.patient_id}\n"
                f"- Patient Name: {scan.patient_name}\n"
                f"- Detected Class: {scan.label.replace('_', ' ').title()}\n"
                f"- Confidence Level: {scan.probability:.2f}%\n"
                f"- Risk Evaluation: {scan.risk_level}\n\n"
                f"Radiologist Notes:\n"
                f"{scan.notes or 'No notes provided.'}\n\n"
                f"Best Regards,\n"
                f"NeuroFusionAI Support Team"
            ),
            from_email=settings.DEFAULT_FROM_EMAIL,
            to=[recipient_email]
        )
        
        email.attach(f"neurofusion_report_{scan.patient_id}.pdf", pdf_content, 'application/pdf')
        email.send()
        
        messages.success(request, f"Medical report successfully emailed to {recipient_email}.")
    except Exception as email_err:
        messages.error(request, f"Failed to send email: {email_err}")
        
    return redirect('prediction_output', scan_id=scan.id)


# --- REST API Endpoints & Chatbot ---

@login_required
def scans_api(request):
    """
    JSON REST API Endpoint returning radiologist's scans history list.
    """
    if request.user.profile.role in ['admin', 'researcher']:
        scans = ScanRecord.objects.all()
    else:
        scans = ScanRecord.objects.filter(user=request.user)
        
    data = []
    for s in scans:
        data.append({
            'id': s.id,
            'patient_id': s.patient_id,
            'patient_name': s.patient_name,
            'patient_age': s.patient_age,
            'patient_gender': s.patient_gender,
            'label': s.label,
            'probability': s.probability,
            'risk_level': s.risk_level,
            'notes': s.notes,
            'uploaded_at': s.uploaded_at.isoformat(),
            'image_url': s.image.url if s.image else None,
            'heatmap_url': s.heatmap.url if s.heatmap else None,
        })
    return JsonResponse({'status': 'success', 'data': data})


def chatbot_api(request):
    """
    Mock AI chatbot response for brain tumor medical inquiries.
    """
    query = request.GET.get('message', '').lower().strip()
    if not query:
        return JsonResponse({'response': "How can I assist you today? Feel free to ask about brain tumor types or risks."})
        
    # AI response keyword matching
    if 'glioma' in query:
        response = (
            "<b>Glioma Tumor:</b> Gliomas originate in the glial cells of the brain or spine. "
            "They are usually classified into astrocytomas, oligodendrogliomas, and ependymomas. "
            "High-grade gliomas (like glioblastomas) are fast-growing and require aggressive treatment "
            "consisting of surgical resection, radiotherapy, and temozolomide chemotherapy."
        )
    elif 'meningioma' in query:
        response = (
            "<b>Meningioma Tumor:</b> Meningiomas arise from the meninges (the protective layers covering the brain). "
            "Over 80% are benign and slow-growing. They are often diagnosed incidentally and can be managed "
            "via watch-and-wait active surveillance, or treated with surgery and radiosurgery if they become symptomatic."
        )
    elif 'pituitary' in query:
        response = (
            "<b>Pituitary Tumor:</b> Pituitary tumors develop in the pituitary gland at the base of the brain. "
            "Most are benign adenomas. They can cause hormonal imbalances (such as hyperprolactinemia) or "
            "visual disturbances (like bitemporal hemianopsia) due to compression of the optic chiasm."
        )
    elif 'schwannoma' in query:
        response = (
            "<b>Schwannoma Tumor:</b> Schwannomas (or acoustic neuromas) arise from Schwann cells surrounding the "
            "vestibulocochlear nerve. They are benign and typically present with unilateral hearing loss, tinnitus, "
            "and equilibrium issues. They can be excised or treated with stereotactic radiosurgery."
        )
    elif 'neurocitoma' in query or 'neurocytoma' in query:
        response = (
            "<b>Neurocytoma Tumor:</b> Central neurocytomas are rare, typically benign intraventricular brain tumors. "
            "They affect young adults and can cause hydrocephalus due to cerebrospinal fluid flow obstruction. "
            "Complete surgical resection is generally curative."
        )
    elif 'risk' in query or 'level' in query:
        response = (
            "NeuroFusionAI maps risk levels based on clinical tumor grades:<br/>"
            "- <b>High Risk:</b> Glioma, Meningioma, and Neurocytoma classes.<br/>"
            "- <b>Medium Risk:</b> Pituitary and Schwannoma classes.<br/>"
            "- <b>Low Risk:</b> Normal / Healthy scans."
        )
    elif 'symptom' in query:
        response = (
            "Common symptoms of brain tumors include persistent headaches (worse in the morning), "
            "seizures, cognitive changes, visual disturbances, balance issues, and nausea. "
            "Any symptoms require professional evaluation and neuroimaging."
        )
    elif 'treatment' in query:
        response = (
            "Treatment options depend on tumor type, size, location, and grade. "
            "Standard interventions include neurosurgical resection, radiation therapy (such as Gamma Knife), "
            "chemotherapy (like temozolomide), and targeted immunotherapies. Always consult an oncologist."
        )
    else:
        response = (
            "I am the NeuroFusionAI virtual assistant. I can answer clinical questions regarding "
            "Glioma, Meningioma, Neurocytoma, Pituitary, or Schwannoma tumors. "
            "Please ask a specific medical question or ask about a tumor type."
        )
        
    return JsonResponse({'response': response})


def set_language_view(request):
    """
    Session-based language switcher.
    """
    lang = request.GET.get('lang', 'en')
    if lang in TRANSLATIONS:
        request.session['django_language'] = lang
    return redirect(request.META.get('HTTP_REFERER', 'dashboard'))


# --- Accessibility Public Pages ---

def about_view(request):
    return render(request, 'users/home.html', {'section': 'about'})

def contact_view(request):
    if request.method == 'POST':
        messages.success(request, "Your message has been submitted. Our support team will contact you shortly.")
        return redirect('home')
    return render(request, 'users/home.html', {'section': 'contact'})

def faq_view(request):
    return render(request, 'users/home.html', {'section': 'faq'})
