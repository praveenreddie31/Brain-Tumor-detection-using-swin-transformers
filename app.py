import streamlit as st
import torch
import timm
import json
from torchvision import transforms
from PIL import Image
import io
import requests

st.set_page_config(
    page_title="Brain Tumor MRI Classifier", 
    page_icon="🧠",
    layout="wide"
)
# ------------------------
# Config
# ------------------------
MODEL_PATH = "best_model.pth"        # Path to your trained weights
LABEL_MAP_PATH = "label_map.json"    # Path to saved label mapping
MODEL_NAME = "swin_tiny_patch4_window7_224"
IMG_SIZE = 224

# Configure Gemini API (Add your API key here)
GEMINI_API_KEY = "AIzaSyDHpMJXQ5-kyVe3tw8ua41PwCm48RHt_Fc"  # Replace with your actual API key
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

# Language options
LANGUAGE_OPTIONS = {
    "English": "English",
    "Hindi": "Hindi", 
    "Tamil": "Tamil",
    "Telugu": "Telugu"
}

# ------------------------
# Load label map
# ------------------------
@st.cache_data
def load_label_map():
    with open(LABEL_MAP_PATH, "r") as f:
        label_map = json.load(f)["idx_to_class"]
        idx_to_class = {int(k): v for k, v in label_map.items()}
    return idx_to_class

try:
    idx_to_class = load_label_map()
    num_classes = len(idx_to_class)
except FileNotFoundError:
    st.error("❌ Label map file not found. Please ensure 'label_map.json' exists.")
    st.stop()

# ------------------------
# Load model
# ------------------------
@st.cache_resource
def load_model():
    try:
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=num_classes)
        ckpt = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found. Please ensure 'best_model.pth' exists.")
        return None

model = load_model()
if model is None:
    st.stop()

# ------------------------
# Preprocessing
# ------------------------
transform = transforms.Compose([
    transforms.Resize(int(IMG_SIZE*1.15)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])

def predict(image: Image.Image):
    """Predict brain tumor class from MRI image"""
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1).numpy()[0]
    return {idx_to_class[i]: float(probs[i]) for i in range(len(idx_to_class))}

def generate_summary(predicted_class, confidence, language, api_key):
    """Generate medical summary using Gemini API"""
    try:
        # Create headers
        headers = {
            'Content-Type': 'application/json',
            'X-goog-api-key': api_key
        }
        
        # Create prompt for summary generation
        prompt = f"""
        You are a medical AI assistant. Provide a brief, informative summary about the brain tumor classification result.
        
        Classification Result: {predicted_class}
        Confidence: {confidence:.1f}%
        
        Please provide:
        1. A brief explanation of what this classification means
        2. General information about this type of brain tumor (if applicable)
        3. Important note that this is AI-generated and should not replace professional medical diagnosis
        
        Respond in {language} language.
        Keep the summary concise (3-4 sentences maximum) and medically accurate but accessible to general audience.
        Include appropriate medical disclaimers.
        """
        
        # Create request payload
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
        }
        
        # Make API request
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                return "Unable to generate summary. No valid response from API."
        else:
            return f"API Error: {response.status_code} - {response.text}"
        
    except Exception as e:
        return f"Unable to generate summary. Error: {str(e)}"

# ------------------------
# Streamlit UI
# ------------------------


# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 1rem 0;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 2rem;
}
.prediction-box {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #667eea;
}
.summary-box {
    background-color: #e8f4fd;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #2196F3;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header"><h1>🧠 Brain Tumor MRI Classifier</h1></div>', unsafe_allow_html=True)

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    selected_language = st.selectbox(
        "Choose Summary Language:",
        options=list(LANGUAGE_OPTIONS.keys()),
        index=0
    )
    
    st.header("📋 Model Information")
    st.info(f"**Model:** {MODEL_NAME}")
    st.info(f"**Classes:** {len(idx_to_class)}")
    st.write("**Available Classes:**")
    for cls in idx_to_class.values():
        st.write(f"• {cls}")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload MRI Scan")
    uploaded_file = st.file_uploader(
        "Choose an MRI image file",
        type=["jpg", "jpeg", "png"],
        help="Upload a brain MRI scan in JPG, JPEG, or PNG format"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded MRI Scan", use_container_width=True)
        
        # Add API key input if not configured
        api_key_to_use = GEMINI_API_KEY
        if GEMINI_API_KEY == "your_gemini_api_key_here":
            st.warning("⚠️ Please enter your Gemini API key to enable summary generation.")
            gemini_key = st.text_input("Enter Gemini API Key:", type="password")
            if gemini_key:
                api_key_to_use = gemini_key

with col2:
    st.header("🔍 Analysis Results")
    
    if uploaded_file is not None:
        # Prediction button
        predict_button = st.button(
            "🚀 Analyze MRI Scan",
            type="primary",
            use_container_width=True,
            help="Click to classify the uploaded MRI scan"
        )
        
        if predict_button:
            with st.spinner("Analyzing MRI scan..."):
                # Make prediction
                preds = predict(image)
                sorted_preds = dict(sorted(preds.items(), key=lambda x: x[1], reverse=True))
                
                # Get top prediction
                top_class = list(sorted_preds.keys())[0]
                top_confidence = list(sorted_preds.values())[0] * 100
                
                # Display prediction results
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.subheader("📊 Classification Results")
                
                # Top prediction highlight
                st.success(f"**Primary Classification:** {top_class} ({top_confidence:.1f}%)")
                
                # All predictions
                st.write("**Detailed Predictions:**")
                for cls, prob in sorted_preds.items():
                    confidence_level = "🔴" if prob < 0.3 else "🟡" if prob < 0.7 else "🟢"
                    st.write(f"{confidence_level} **{cls}**: {prob*100:.2f}%")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Prediction confidence chart
                st.subheader("📈 Confidence Distribution")
                st.bar_chart(sorted_preds)
                
                # Generate summary
                st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                st.subheader(f"📝 Medical Summary ({selected_language})")
                
                with st.spinner(f"Generating summary in {selected_language}..."):
                    try:
                        summary = generate_summary(
                            top_class, 
                            top_confidence, 
                            LANGUAGE_OPTIONS[selected_language],
                            api_key_to_use
                        )
                        st.write(summary)
                    except Exception as e:
                        st.error(f"Failed to generate summary: {str(e)}")
                        st.write("Please check your Gemini API key and internet connection.")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Medical disclaimer
                st.warning("""
                ⚠️ **Important Medical Disclaimer:** 
                This AI classification is for educational/research purposes only and should NOT be used for medical diagnosis. 
                Always consult qualified healthcare professionals for medical advice and diagnosis.
                """)
    else:
        st.info("👆 Please upload an MRI scan to begin analysis")
        
        # Show example of what the analysis will include
        st.subheader("📋 What You'll Get:")
        st.write("✅ **Classification Results** - Detailed probability scores for each tumor type")
        st.write("✅ **Visual Chart** - Easy-to-read confidence distribution")
        st.write(f"✅ **Medical Summary** - AI-generated explanation in {selected_language}")
        st.write("✅ **Safety Disclaimers** - Important medical guidance")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🔬 Powered by Deep Learning | 🌐 Multi-language Support | 🛡️ Privacy-First Design</p>
</div>
""", unsafe_allow_html=True)

# Instructions
with st.expander("📖 How to Use"):
    st.write("""
    1. **Upload Image**: Select an MRI brain scan (JPG, PNG formats supported)
    2. **Choose Language**: Select your preferred language for the summary from the sidebar
    3. **Analyze**: Click the 'Analyze MRI Scan' button to get predictions
    4. **Review Results**: Check the classification results and confidence scores
    5. **Read Summary**: Get an AI-generated explanation in your chosen language
    
    **Note**: Make sure to configure your Gemini API key for summary generation.
    """)

with st.expander("⚙️ Setup Instructions"):
    st.write("""
    **Required Files:**
    - `best_model.pth` - Your trained model weights
    - `label_map.json` - Class label mapping file
    
    **API Configuration:**
    - Get a free Gemini API key from Google AI Studio
    - Replace `your_gemini_api_key_here` in the code with your actual key
    - Or enter it in the text input when prompted
    
    **Dependencies:**
    ```bash
    pip install streamlit torch timm torchvision pillow requests
    ```
    """)