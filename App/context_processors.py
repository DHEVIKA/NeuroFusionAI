from .translation_data import TRANSLATIONS

def translation_processor(request):
    # Retrieve language choice from session, default to English
    lang = request.session.get('django_language', 'en')
    if lang not in TRANSLATIONS:
        lang = 'en'
        
    return {
        't': TRANSLATIONS[lang],
        'current_lang': lang
    }
