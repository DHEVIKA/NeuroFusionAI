from django.urls import path
from . import views

urlpatterns = [
    # General / Public routes
    path('', views.home_view, name='home'),
    path('about/', views.about_view, name='about'),
    path('contact/', views.contact_view, name='contact'),
    path('faq/', views.faq_view, name='faq'),
    path('set-language/', views.set_language_view, name='set_language'),

    # Authentication routes
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('profile/', views.profile_view, name='profile'),
    path('change-password/', views.change_password_view, name='change_password'),
    path('api/google-login/', views.google_login_api, name='google_login_api'),

    # App core routes
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('upload/', views.upload_scan_view, name='upload_scan'),
    path('prediction/<int:scan_id>/', views.prediction_output_view, name='prediction_output'),
    path('database/', views.database_view, name='database'),
    path('delete/<int:scan_id>/', views.delete_scan_view, name='delete_scan'),
    path('analytics/', views.analytics_view, name='analytics'),

    # Exporters & Communication routes
    path('download-report/<int:scan_id>/', views.download_report_view, name='download_report'),
    path('email-report/<int:scan_id>/', views.email_report_view, name='email_report'),

    # REST APIs
    path('api/scans/', views.scans_api, name='scans_api'),
    path('api/chatbot/', views.chatbot_api, name='chatbot_api'),
]
