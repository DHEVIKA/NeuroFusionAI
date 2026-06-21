from django.db import models
from django.contrib.auth.models import User

class Profile(models.Model):
    ROLE_CHOICES = [
        ('admin', 'Administrator'),
        ('radiologist', 'Radiologist'),
        ('doctor', 'Medical Doctor'),
        ('researcher', 'Researcher'),
        ('public', 'Public User'),
    ]
    
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='public')
    professional_id = models.CharField(max_length=50, blank=True, null=True)
    department = models.CharField(max_length=100, blank=True, null=True)
    avatar = models.ImageField(upload_to='avatars/', blank=True, null=True)
    google_signed_in = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.user.username} - {self.get_role_display()}"


class ScanRecord(models.Model):
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other'),
    ]

    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='scans')
    image = models.ImageField(upload_to='scans/')
    heatmap = models.ImageField(upload_to='scans/', blank=True, null=True)
    
    # Prediction Results
    label = models.CharField(max_length=100)
    probability = models.FloatField()  # percentage value or fraction (will store percentage 0.0 - 100.0)
    risk_level = models.CharField(max_length=20, default='Low')  # Low, Medium, High
    
    # Patient Demographics
    patient_name = models.CharField(max_length=100)
    patient_id = models.CharField(max_length=50)
    patient_age = models.IntegerField(default=30)
    patient_gender = models.CharField(max_length=10, choices=GENDER_CHOICES, default='M')
    
    # Doctor Comments
    notes = models.TextField(blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self):
        return f"Scan {self.patient_id} - {self.label} ({self.probability:.2f}%)"
