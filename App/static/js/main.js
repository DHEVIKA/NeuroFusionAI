// NeuroFusionAI JavaScript - Theme, Accessibility, Drag-Drop, and Voice Assist

document.addEventListener('DOMContentLoaded', function() {
    
    // --- 1. Theme Switcher ---
    const themeToggleBtn = document.getElementById('theme-toggle');
    if (themeToggleBtn) {
        // Load initial theme
        const savedTheme = localStorage.getItem('theme') || 'dark';
        if (savedTheme === 'light') {
            document.body.classList.add('light-theme');
            themeToggleBtn.innerHTML = '<i class="bi bi-moon-stars-fill"></i>';
        } else {
            themeToggleBtn.innerHTML = '<i class="bi bi-sun-fill"></i>';
        }
        
        themeToggleBtn.addEventListener('click', function() {
            document.body.classList.toggle('light-theme');
            const isLight = document.body.classList.contains('light-theme');
            localStorage.setItem('theme', isLight ? 'light' : 'dark');
            themeToggleBtn.innerHTML = isLight 
                ? '<i class="bi bi-moon-stars-fill"></i>' 
                : '<i class="bi bi-sun-fill"></i>';
        });
    }

    // --- 2. Web Speech API Narration ---
    const voiceAssistBtn = document.getElementById('voice-assist');
    if (voiceAssistBtn) {
        voiceAssistBtn.addEventListener('click', function() {
            const labelEl = document.getElementById('result-label');
            const confEl = document.getElementById('result-confidence');
            const riskEl = document.getElementById('result-risk');
            const patientEl = document.getElementById('result-patient');
            
            if (labelEl && confEl && riskEl) {
                const label = labelEl.innerText.replace(/_/g, ' ');
                const confidence = confEl.innerText;
                const risk = riskEl.innerText;
                const patientName = patientEl ? patientEl.innerText : 'Unknown Patient';
                
                const textToSpeak = `MRI Scan diagnosis completed for patient ${patientName}. The neural network predicted ${label} with a confidence score of ${confidence}, yielding a ${risk} risk evaluation. Please inspect the gradient class activation mappings for localization.`;
                
                // Trigger Web Speech Synthesis
                if ('speechSynthesis' in window) {
                    // Cancel active speakings
                    window.speechSynthesis.cancel();
                    const utterance = new SpeechSynthesisUtterance(textToSpeak);
                    utterance.rate = 0.95; // slightly slower for clarity
                    utterance.pitch = 1.0;
                    window.speechSynthesis.speak(utterance);
                    
                    // Button feedback animation
                    voiceAssistBtn.classList.add('btn-success');
                    voiceAssistBtn.classList.remove('btn-outline-primary');
                    setTimeout(() => {
                        voiceAssistBtn.classList.remove('btn-success');
                        voiceAssistBtn.classList.add('btn-outline-primary');
                    }, 1500);
                } else {
                    alert("Text-to-speech is not supported by your browser.");
                }
            }
        });
    }

    // --- 3. Drag and Drop File Upload & Image Preview ---
    const dragArea = document.getElementById('drag-drop-area');
    const fileInput = document.getElementById('mri-file-input');
    const previewContainer = document.getElementById('mri-preview-container');
    const previewImage = document.getElementById('mri-preview-img');
    const placeholderText = document.getElementById('upload-placeholder');
    const uploadForm = document.getElementById('upload-scan-form');
    const submitBtn = document.getElementById('submit-scan-btn');

    if (dragArea && fileInput) {
        // Trigger click on input when clicking drag area
        dragArea.addEventListener('click', () => fileInput.click());

        // Visual drag states
        ['dragenter', 'dragover'].forEach(eventName => {
            dragArea.addEventListener(eventName, (e) => {
                e.preventDefault();
                dragArea.classList.add('dragover');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dragArea.addEventListener(eventName, (e) => {
                e.preventDefault();
                dragArea.classList.remove('dragover');
            }, false);
        });

        // Handle dropped files
        dragArea.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length) {
                fileInput.files = files;
                handleFilePreview(files[0]);
            }
        });

        // Handle selected files
        fileInput.addEventListener('change', (e) => {
            if (fileInput.files.length) {
                handleFilePreview(fileInput.files[0]);
            }
        });
    }

    function handleFilePreview(file) {
        if (file && file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onloadend = function() {
                previewImage.src = reader.result;
                previewContainer.style.display = 'block';
                placeholderText.style.display = 'none';
            }
        }
    }

    // --- 4. Scanner Simulation on Submit ---
    if (uploadForm && submitBtn) {
        uploadForm.addEventListener('submit', function() {
            // Add scanning animation overlay
            const scanOverlay = document.getElementById('scanner-overlay');
            if (scanOverlay) {
                scanOverlay.classList.add('scanline-active');
            }
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Executing Neural Network...';
        });
    }

    // --- 5. AI Chatbot Interface ---
    const chatBubble = document.getElementById('chatbot-bubble');
    const chatWindow = document.getElementById('chatbot-window');
    const closeChatBtn = document.getElementById('close-chatbot');
    const sendChatBtn = document.getElementById('send-chat');
    const chatInput = document.getElementById('chat-input');
    const chatBody = document.getElementById('chatbot-body');

    if (chatBubble && chatWindow) {
        // Toggle window
        chatBubble.addEventListener('click', () => {
            chatWindow.style.display = 'flex';
            chatBubble.style.display = 'none';
        });

        if (closeChatBtn) {
            closeChatBtn.addEventListener('click', () => {
                chatWindow.style.display = 'none';
                chatBubble.style.display = 'flex';
            });
        }

        // Send message function
        const sendMessage = function() {
            const text = chatInput.value.trim();
            if (!text) return;

            // Append user message
            appendMessage(text, 'user');
            chatInput.value = '';

            // Loading state message
            const loaderId = 'bot-loader-' + Date.now();
            appendMessage('<span class="spinner-grow spinner-grow-sm" role="status"></span> Medical Assistant is thinking...', 'bot', loaderId);

            // Fetch API response
            fetch(`/api/chatbot/?message=${encodeURIComponent(text)}`)
                .then(res => res.json())
                .then(data => {
                    // Remove loader and append bot response
                    const loader = document.getElementById(loaderId);
                    if (loader) loader.remove();
                    appendMessage(data.response, 'bot');
                })
                .catch(err => {
                    const loader = document.getElementById(loaderId);
                    if (loader) loader.remove();
                    appendMessage('Sorry, chatbot services are currently experiencing issues. Please try again.', 'bot');
                });
        };

        if (sendChatBtn) {
            sendChatBtn.addEventListener('click', sendMessage);
        }

        if (chatInput) {
            chatInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
        }
    }

    function appendMessage(htmlContent, sender, id = '') {
        const msgDiv = document.createElement('div');
        msgDiv.className = `chat-msg ${sender}`;
        if (id) msgDiv.id = id;
        msgDiv.innerHTML = htmlContent;
        chatBody.appendChild(msgDiv);
        chatBody.scrollTop = chatBody.scrollHeight;
    }
    
    // --- 6. Google Sign-In Integrator (Mock) ---
    const googleLoginBtn = document.getElementById('google-signin-btn');
    if (googleLoginBtn) {
        googleLoginBtn.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Get CSRF Token
            const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]').value;
            
            fetch('/api/google-login/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken
                }
            })
            .then(res => res.json())
            .then(data => {
                if (data.status === 'success') {
                    window.location.href = data.redirect_url;
                }
            })
            .catch(err => console.error("Google authentication error:", err));
        });
    }
});
