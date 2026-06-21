from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from .models import Profile, ScanRecord

class RadiologistRegisterForm(forms.ModelForm):
    username = forms.CharField(max_length=150, required=True, widget=forms.TextInput(attrs={
        'class': 'form-control bg-dark text-white border-secondary',
        'placeholder': 'Enter username'
    }))
    email = forms.EmailField(required=True, widget=forms.EmailInput(attrs={
        'class': 'form-control bg-dark text-white border-secondary',
        'placeholder': 'Enter email address'
    }))
    password = forms.CharField(widget=forms.PasswordInput(attrs={
        'class': 'form-control bg-dark text-white border-secondary',
        'placeholder': 'Enter password'
    }))
    confirm_password = forms.CharField(widget=forms.PasswordInput(attrs={
        'class': 'form-control bg-dark text-white border-secondary',
        'placeholder': 'Confirm password'
    }))
    role = forms.ChoiceField(choices=Profile.ROLE_CHOICES, required=True, widget=forms.Select(attrs={
        'class': 'form-select bg-dark text-white border-secondary'
    }))
    professional_id = forms.CharField(required=False, widget=forms.TextInput(attrs={
        'class': 'form-control bg-dark text-white border-secondary',
        'placeholder': 'Medical license or ID number (optional)'
    }))
    department = forms.CharField(required=False, widget=forms.TextInput(attrs={
        'class': 'form-control bg-dark text-white border-secondary',
        'placeholder': 'Department (e.g. Radiology, Neurology)'
    }))

    class Meta:
        model = User
        fields = ['username', 'email', 'password']

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get("password")
        confirm_password = cleaned_data.get("confirm_password")
        if password != confirm_password:
            raise forms.ValidationError("Passwords do not match.")
        return cleaned_data


class ProfileUpdateForm(forms.ModelForm):
    first_name = forms.CharField(max_length=30, required=False, widget=forms.TextInput(attrs={
        'class': 'form-control bg-dark text-white border-secondary'
    }))
    last_name = forms.CharField(max_length=30, required=False, widget=forms.TextInput(attrs={
        'class': 'form-control bg-dark text-white border-secondary'
    }))
    email = forms.EmailField(required=True, widget=forms.EmailInput(attrs={
        'class': 'form-control bg-dark text-white border-secondary'
    }))

    class Meta:
        model = Profile
        fields = ['role', 'professional_id', 'department', 'avatar']
        widgets = {
            'role': forms.Select(attrs={'class': 'form-select bg-dark text-white border-secondary'}),
            'professional_id': forms.TextInput(attrs={'class': 'form-control bg-dark text-white border-secondary'}),
            'department': forms.TextInput(attrs={'class': 'form-control bg-dark text-white border-secondary'}),
            'avatar': forms.FileInput(attrs={'class': 'form-control bg-dark text-white border-secondary'}),
        }


class ScanUploadForm(forms.ModelForm):
    class Meta:
        model = ScanRecord
        fields = ['image', 'patient_name', 'patient_id', 'patient_age', 'patient_gender', 'notes']
        widgets = {
            'image': forms.FileInput(attrs={'class': 'form-control bg-dark text-white border-secondary', 'id': 'mri-file-input'}),
            'patient_name': forms.TextInput(attrs={'class': 'form-control bg-dark text-white border-secondary', 'placeholder': 'Patient Full Name'}),
            'patient_id': forms.TextInput(attrs={'class': 'form-control bg-dark text-white border-secondary', 'placeholder': 'MRN-XXXXX'}),
            'patient_age': forms.NumberInput(attrs={'class': 'form-control bg-dark text-white border-secondary', 'placeholder': 'Age'}),
            'patient_gender': forms.Select(attrs={'class': 'form-select bg-dark text-white border-secondary'}),
            'notes': forms.Textarea(attrs={'class': 'form-control bg-dark text-white border-secondary', 'rows': 4, 'placeholder': 'Enter observations here...'}),
        }
