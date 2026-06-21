from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('App.urls')),
]

# Serve media files in development (when not using Cloudinary)
if settings.DEBUG and not getattr(settings, 'CLOUDINARY_URL', None):
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
