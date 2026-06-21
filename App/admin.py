from django.contrib import admin
from .models import Profile, ScanRecord

@admin.register(Profile)
class ProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'role', 'department', 'google_signed_in')
    list_filter = ('role', 'google_signed_in')
    search_fields = ('user__username', 'department', 'professional_id')

@admin.register(ScanRecord)
class ScanRecordAdmin(admin.ModelAdmin):
    list_display = ('patient_id', 'patient_name', 'label', 'probability', 'risk_level', 'uploaded_at')
    list_filter = ('label', 'risk_level', 'patient_gender')
    search_fields = ('patient_name', 'patient_id', 'label')
    ordering = ('-uploaded_at',)
