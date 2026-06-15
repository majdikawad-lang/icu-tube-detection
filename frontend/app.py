import streamlit as st
import requests
import base64
from io import BytesIO
from PIL import Image

# Configure page
st.set_page_config(
    page_title="ICU Tube Displacement Monitor",
    page_icon="🏥",
    layout="wide"
)

# Constants
API_URL = "http://localhost:8000/predict"

def main():
    st.title("🏥 ICU Tube Displacement Detection System")
    st.markdown("""
    Upload a patient's chest X-ray in DICOM format to analyze the position of the Endotracheal Tube.
    The AI model will detect the tube tip and the carina to determine if the tube is safely positioned.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a DICOM file (.dcm)", type=["dcm"])
    
    if uploaded_file is not None:
        st.info(f"File uploaded: {uploaded_file.name}")
        
        if st.button("Analyze Image"):
            with st.spinner("Processing DICOM and running AI inference..."):
                try:
                    # Send file to FastAPI backend
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/dicom")}
                    response = requests.post(API_URL, files=files)
                    
                    if response.status_code == 200:
                        results = response.json()
                        display_results(results)
                    else:
                        st.error(f"Error from server: {response.text}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("Failed to connect to the backend server. Make sure FastAPI is running on port 8000.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

def display_results(results):
    st.markdown("---")
    st.subheader("Analysis Results")
    
    col1, col2 = st.columns([1, 1])
    
    status = results["status"]
    confidence = results["confidence"]
    distance = results["distance_px"]
    
    with col1:
        st.markdown("### Clinical Status")
        
        if status == "Safe":
            st.success(f"**STATUS: {status}**")
            # CSS for green glowing box
            st.markdown(
                """
                <div style="background-color:#d4edda; color:#155724; padding:20px; border-radius:10px; text-align:center; border: 2px solid #28a745; box-shadow: 0 0 15px #28a745;">
                    <h2 style="margin:0;">✅ TUBE POSITION SAFE</h2>
                </div>
                """, unsafe_allow_html=True
            )
        else:
            st.error(f"**STATUS: {status}**")
            # CSS for red flashing box
            st.markdown(
                """
                <style>
                @keyframes blink {
                  50% { opacity: 0.5; box-shadow: 0 0 30px #dc3545; }
                }
                .alert-box {
                  background-color: #f8d7da;
                  color: #721c24;
                  padding: 20px;
                  border-radius: 10px;
                  text-align: center;
                  border: 2px solid #dc3545;
                  animation: blink 1s linear infinite;
                  box-shadow: 0 0 15px #dc3545;
                }
                </style>
                <div class="alert-box">
                    <h2 style="margin:0;">🚨 DISPLACEMENT DETECTED 🚨</h2>
                </div>
                """, unsafe_allow_html=True
            )
            
        st.markdown("### Metrics")
        st.metric("AI Confidence", f"{confidence * 100:.1f}%")
        st.metric("Distance (Tip to Carina)", f"{distance:.1f} px")
        
        st.markdown("""
        **Note:** 
        - Optimal distance is generally 3-7 cm. 
        - This tool is for investigational use and should not replace clinical judgement.
        """)

    with col2:
        st.markdown("### Annotated X-Ray")
        # Decode base64 image
        image_data = base64.b64decode(results["image_base64"])
        image = Image.open(BytesIO(image_data))
        st.image(image, caption="AI Detected Landmarks (Red/Blue: Tube Tip, Green: Carina)", use_container_width=True)

if __name__ == "__main__":
    main()
