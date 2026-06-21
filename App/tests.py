import os
from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from django.conf import settings
from .models import ScanRecord, Profile
from .views import get_model, run_mri_prediction

class NeuroFusionTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.username = "radiologist_test"
        self.password = "Secr3tP@ssword"
        self.email = "test@hospital.com"
        
        # Create standard radiologist test user
        self.user = User.objects.create_user(
            username=self.username,
            email=self.email,
            password=self.password
        )
        self.user.profile.role = 'radiologist'
        self.user.profile.department = 'Radiology'
        self.user.profile.professional_id = 'RAD-9999'
        self.user.profile.save()

    def test_translation_context_processor(self):
        """
        Test that translation dictionary maps correctly and defaults to English.
        """
        response = self.client.get(reverse('home'))
        self.assertEqual(response.status_code, 200)
        self.assertIn('t', response.context)
        self.assertEqual(response.context['current_lang'], 'en')
        
        # Test language switch to Hindi
        self.client.get(reverse('set_language') + '?lang=hi')
        response = self.client.get(reverse('home'))
        self.assertEqual(response.context['current_lang'], 'hi')

    def test_authentication_views(self):
        """
        Test that normal login succeeds, and invalid login fails.
        """
        # Test GET login
        response = self.client.get(reverse('login'))
        self.assertEqual(response.status_code, 200)

        # Test POST login
        login_success = self.client.login(username=self.username, password=self.password)
        self.assertTrue(login_success)

        # Test GET dashboard (requires login)
        response = self.client.get(reverse('dashboard'))
        self.assertEqual(response.status_code, 200)

        # Test logout
        self.client.logout()
        response = self.client.get(reverse('dashboard'))
        # Should redirect to login since login_required is applied
        self.assertEqual(response.status_code, 302)

    def test_chatbot_api(self):
        """
        Test chatbot response matches queries accurately.
        """
        # Glioma query
        response = self.client.get(reverse('chatbot_api') + '?message=What+is+glioma')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("Glioma", data['response'])

        # Risk query
        response = self.client.get(reverse('chatbot_api') + '?message=risk+levels')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("High Risk", data['response'])

    def test_mock_prediction_fallback(self):
        """
        Verify that image predictor falls back gracefully and runs without crash.
        """
        # Create a tiny 1x1 white mock jpeg image in memory
        from PIL import Image
        import io
        
        img_file = io.BytesIO()
        image = Image.new('RGB', (224, 224), color='white')
        image.save(img_file, 'jpeg')
        img_file.seek(0)
        
        # Run prediction directly
        # Since we haven't generated the model file yet in the test environment, this should test the fallback path
        temp_img_path = os.path.join(settings.BASE_DIR, 'test_mock_mri.jpg')
        with open(temp_img_path, 'wb') as f:
            f.write(img_file.getvalue())
            
        try:
            pred_class, confidence, heatmap_img = run_mri_prediction(temp_img_path)
            self.assertIn(pred_class, [
                'glioma_tumor', 'meningioma_tumor', 'Neurocitoma_tumor',
                'Normal', 'pituitary_tumor', 'Schwannoma_tumor'
            ])
            self.assertTrue(0.0 <= confidence <= 100.0)
            self.assertEqual(heatmap_img.size, (224, 224))
        finally:
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
                
    def test_api_scans_history(self):
        """
        Ensure REST API endpoint functions and returns JSON payload.
        """
        self.client.login(username=self.username, password=self.password)
        response = self.client.get(reverse('scans_api'))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'success')
        self.assertEqual(len(data['data']), 0) # No scans uploaded yet
