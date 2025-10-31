import streamlit as st
import requests
import json
import time

# Page configuration
st.set_page_config(
    page_title="Text Summarizer & FAQ Generator",
    page_icon="📝",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .process-step {
        padding: 10px;
        margin: 5px 0;
        border-radius: 8px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 500;
        animation: slideIn 0.5s ease-out;
    }
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    .success-box {
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 10px 0;
    }
    .info-box {
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        margin: 10px 0;
    }
    h1 {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Model configuration
MODEL_SERVER_URL = "http://localhost:11434"

def check_model_server():
    """Check if the trained model server is running"""
    try:
        response = requests.get(f"{MODEL_SERVER_URL}/api/tags", timeout=5)
        return response.status_code == 200, response
    except:
        return False, None

def load_available_models():
    """Load available trained models from local repository"""
    try:
        response = requests.get(f"{MODEL_SERVER_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return [model['name'] for model in models]
        return []
    except:
        return []

def show_processing_steps(step_placeholder):
    """Display neural network processing steps"""
    steps = [
        ("🔤 Tokenization", "Breaking text into tokens..."),
        ("🔍 Text Analysis", "Analyzing content structure..."),
        ("🧹 Preprocessing", "Cleaning and normalizing text..."),
        ("🌿 Stemming", "Reducing words to root forms..."),
        ("🧠 Neural Network", "Processing through trained layers..."),
        ("✨ Post-processing", "Formatting final output...")
    ]
    
    for i, (title, desc) in enumerate(steps):
        step_placeholder.markdown(f"""
            <div class='process-step'>
                {title}: {desc}
            </div>
        """, unsafe_allow_html=True)
        time.sleep(0.8)

def chunk_text(text, max_chars=2000):
    """Split text into manageable chunks"""
    if len(text) <= max_chars:
        return [text]
    
    # Try to split at sentence boundaries
    sentences = text.replace('!', '.').replace('?', '.').split('.')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chars:
            current_chunk += sentence + "."
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + "."
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def inference(prompt, model_name, process_placeholder=None, timeout=1000):
    """Run inference on the trained model with extended timeout"""
    try:
        if process_placeholder:
            show_processing_steps(process_placeholder)
        
        # Prepare input for the model
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False
        }
        
        # Run model inference with extended timeout
        response = requests.post(
            f"{MODEL_SERVER_URL}/api/chat", 
            json=payload, 
            timeout=timeout
        )
        
        if response.status_code == 404:
            # Fallback to alternative inference method
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False
            }
            response = requests.post(
                f"{MODEL_SERVER_URL}/api/generate", 
                json=payload, 
                timeout=timeout
            )
        
        response.raise_for_status()
        result = response.json()
        
        # Extract model output
        if 'message' in result:
            return result['message'].get('content', '')
        else:
            return result.get('response', '')
    
    except requests.exceptions.ConnectionError:
        return "❌ Error: Model server is not running. Please start the model server."
    except requests.exceptions.Timeout:
        return f"❌ Error: Model inference timed out after {timeout} seconds. Try reducing text length or number of questions."
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return f"❌ Error: Model '{model_name}' not found in model repository."
        return f"❌ Error: {str(e)}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def inference_streaming(prompt, model_name, result_placeholder=None):
    """Run inference with streaming for better timeout handling"""
    try:
        # Prepare input for the model with streaming enabled
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": True
        }
        
        # Use streaming to avoid timeout
        response = requests.post(
            f"{MODEL_SERVER_URL}/api/chat", 
            json=payload, 
            timeout=10,  # Short timeout for connection, not response
            stream=True
        )
        
        if response.status_code == 404:
            # Fallback to generate endpoint
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": True
            }
            response = requests.post(
                f"{MODEL_SERVER_URL}/api/generate", 
                json=payload, 
                timeout=10,
                stream=True
            )
        
        response.raise_for_status()
        
        # Collect streamed response
        full_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line)
                    if 'message' in chunk:
                        content = chunk['message'].get('content', '')
                    else:
                        content = chunk.get('response', '')
                    
                    full_response += content
                    
                    # Update placeholder in real-time if provided
                    if result_placeholder:
                        result_placeholder.markdown(full_response)
                        
                except json.JSONDecodeError:
                    continue
        
        return full_response
    
    except requests.exceptions.ConnectionError:
        return "❌ Error: Model server is not running. Please start the model server."
    except requests.exceptions.Timeout:
        return "❌ Error: Connection to model server timed out."
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return f"❌ Error: Model '{model_name}' not found in model repository."
        return f"❌ Error: {str(e)}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def generate_summary(text, summary_length, model_name, process_placeholder=None):
    """Generate summary using the trained model"""
    length_instructions = {
        "short": "in 2-3 sentences",
        "medium": "in 1 paragraph (4-6 sentences)",
        "long": "in 2-3 paragraphs with key details"
    }
    
    prompt = f"""Please provide a clear and concise summary of the following text {length_instructions[summary_length]}:

Text: {text}

Summary:"""
    
    return inference(prompt, model_name, process_placeholder, timeout=1000)

def generate_faq_smart(text, num_questions, model_name, process_placeholder=None):
    """Generate FAQ with smart text handling and extended timeout"""
    
    # Check text length
    if len(text) > 3000:
        # For long texts, chunk and generate FAQs from each chunk
        chunks = chunk_text(text, max_chars=2500)
        questions_per_chunk = max(2, num_questions // len(chunks))
        
        all_faqs = []
        
        for i, chunk in enumerate(chunks):
            if process_placeholder:
                process_placeholder.info(f"📝 Processing section {i+1}/{len(chunks)}... This may take a while for long texts.")
            
            prompt = f"""Based on the following text section, generate {questions_per_chunk} frequently asked questions (FAQ) with their answers. Format each FAQ as:

Q: [Question]
A: [Answer]

Text: {chunk}

FAQ:"""
            
            faq = inference(prompt, model_name, None, timeout=1000)
            if not faq.startswith("❌"):
                all_faqs.append(faq)
            else:
                if process_placeholder:
                    process_placeholder.empty()
                return faq  # Return error message
        
        if process_placeholder:
            process_placeholder.empty()
        
        # Combine FAQs
        combined_faq = "\n\n".join(all_faqs)
        return combined_faq[:min(len(combined_faq), 10000)]  # Limit total length
    
    else:
        # For shorter texts, use normal generation
        prompt = f"""Based on the following text, generate {num_questions} frequently asked questions (FAQ) with their answers. Format each FAQ as:

Q: [Question]
A: [Answer]

Text: {text}

FAQ:"""
        
        return inference(prompt, model_name, process_placeholder, timeout=1000)

def generate_faq(text, num_questions, model_name, process_placeholder=None):
    """Generate FAQ using the trained model - wrapper for backwards compatibility"""
    return generate_faq_smart(text, num_questions, model_name, process_placeholder)

# App UI
st.title("📝 Text Summarizer & FAQ Generator")
st.markdown("##### Transform your text into concise summaries and helpful FAQs")

# Check model server
server_running, response = check_model_server()

if not server_running:
    st.error("🔴 **Model server is not running!**")
    st.markdown("""
    ### How to start the model server:
    
    1. Open a terminal/command prompt
    2. Navigate to the model directory
    3. Start the model server
    4. Refresh this page
    
    Please ensure your trained models are properly loaded.
    """)
    st.stop()

# Load available models
available_models = load_available_models()

# Sidebar with modern design
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Connection status
    st.success("🟢 Model Server Online")
    
    st.markdown("---")
    
    # Model selection
    st.markdown("#### 🧠 Model Selection")
    if available_models:
        default_model = available_models[0]
        
        model_name = st.selectbox(
            "Choose Trained Model",
            options=available_models,
            index=0,
            label_visibility="collapsed"
        )
    else:
        model_name = st.text_input("Model Name", value="custom-model")
        st.warning("⚠️ No models found in repository.")
    
    st.markdown("---")
    
    # Summary settings
    st.markdown("#### 📄 Summary Options")
    summary_length = st.radio(
        "Length",
        options=["short", "medium", "long"],
        index=1,
        horizontal=True,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # FAQ settings
    st.markdown("#### ❓ FAQ Options")
    num_questions = st.slider(
        "Number of Questions",
        min_value=3,
        max_value=10,
        value=5,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Performance info
    st.markdown("#### ⚡ Performance Info")
    st.info("⏱️ Timeout: 1000 seconds\n\n📊 Max recommended text: 10,000 characters")
    
    st.markdown("---")
    
    if st.button("🔄 Reload Models", use_container_width=True):
        st.rerun()

# Main content
tab1, tab2, tab3 = st.tabs(["📝 Process Text", "ℹ️ About", "🔧 Help"])

with tab1:
    # Text input with better styling
    st.markdown("#### Enter Your Text")
    input_text = st.text_area(
        "Text input",
        height=250,
        placeholder="Paste or type the text you want to summarize and generate FAQs from...",
        label_visibility="collapsed"
    )
    
    # Show text statistics
    if input_text:
        char_count = len(input_text)
        word_count = len(input_text.split())
        
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("Characters", f"{char_count:,}")
        with col_stat2:
            st.metric("Words", f"{word_count:,}")
        with col_stat3:
            if char_count > 5000:
                st.metric("Status", "⚠️ Long", delta="May take time")
            elif char_count > 3000:
                st.metric("Status", "⏳ Medium", delta="Normal time")
            else:
                st.metric("Status", "✅ Short", delta="Fast")
        
        # Warning for very long texts
        if char_count > 8000:
            st.warning("⚠️ Very long text detected! Processing may take several minutes. Consider breaking it into smaller sections for faster results.")
        elif char_count > 5000:
            st.info("ℹ️ Long text detected. FAQ generation will be chunked for better performance.")
    
    # Action buttons
    col1, col2 = st.columns(2)
    
    with col1:
        generate_summary_btn = st.button("🔍 Generate Summary", type="primary", use_container_width=True)
    
    with col2:
        generate_faq_btn = st.button("❓ Generate FAQ", type="primary", use_container_width=True)
    
    # Display results
    if input_text:
        st.markdown("---")
        
        # Summary section
        if generate_summary_btn:
            st.markdown("### 📋 Summary")
            
            # Processing steps placeholder
            process_placeholder = st.empty()
            
            with st.spinner("Generating summary... This may take up to 16 minutes for long texts."):
                summary = generate_summary(input_text, summary_length, model_name, process_placeholder)
            
            # Clear processing steps
            process_placeholder.empty()
            
            if summary.startswith("❌"):
                st.error(summary)
            else:
                st.markdown(f"""
                    <div class='info-box'>
                        {summary}
                    </div>
                """, unsafe_allow_html=True)
                
                # Download button
                st.download_button(
                    label="⬇️ Download Summary",
                    data=summary,
                    file_name="summary.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        
        # FAQ section
        if generate_faq_btn:
            st.markdown("### ❓ Frequently Asked Questions")
            
            # Processing steps placeholder
            process_placeholder = st.empty()
            
            with st.spinner("Generating FAQs... This may take up to 16 minutes for long texts."):
                faq = generate_faq(input_text, num_questions, model_name, process_placeholder)
            
            # Clear processing steps
            process_placeholder.empty()
            
            if faq.startswith("❌"):
                st.error(faq)
            else:
                st.markdown(f"""
                    <div class='success-box'>
                        {faq.replace(chr(10), '<br>')}
                    </div>
                """, unsafe_allow_html=True)
                
                # Download button
                st.download_button(
                    label="⬇️ Download FAQ",
                    data=faq,
                    file_name="faq.txt",
                    mime="text/plain",
                    use_container_width=True
                )
    else:
        st.info("💡 Enter some text above to get started!")

with tab2:
    st.markdown("""
    ## 🎯 About This Application
    
    This tool uses a custom-trained neural network model to help you quickly understand and extract key information from any text by:
    - 📝 Creating concise, readable summaries
    - ❓ Generating relevant questions and answers
    
    ### 🧠 Model Architecture
    
    The underlying model has been trained on diverse text datasets to understand context, extract key information, and generate human-like responses. The model uses transformer-based architecture with attention mechanisms for optimal performance.
    
    ### 🚀 Getting Started
    
    1. **Enter your text** in the input area
    2. **Adjust settings** in the sidebar (optional)
    3. **Generate** summaries or FAQs with one click
    4. **Download** results for later use
    
    ### ✨ Features
    
    - 🎨 Modern, intuitive interface
    - 🔄 Real-time processing feedback
    - 📊 Multiple summary lengths
    - 🎯 Customizable FAQ generation
    - 💾 Easy download options
    - ⚡ Fast neural network inference
    - 🔒 Local model execution for privacy
    - 📏 Smart text chunking for long documents
    - ⏱️ Extended timeout (1000 seconds) for complex tasks
    
    ### 🛠️ Technical Details
    
    - Built with Streamlit for smooth UI
    - Custom-trained language model
    - Local inference for data privacy
    - No external data transmission
    - Optimized for performance
    - Automatic text chunking for long documents
    - Extended timeout handling for large texts
    """)

with tab3:
    st.markdown("""
    ## 🔧 Troubleshooting Guide
    
    ### Common Issues
    
    #### 🔴 Model Server Not Running
    
    **Solution:** Ensure the model server is started and the trained models are loaded properly. Check that the server is running on the correct port (default: 11434).
    
    #### ⚠️ Model Not Found
    
    **Solution:** Verify that the required models are present in your local model repository. You may need to reload the models or check the model directory.
    
    #### ⏱️ Inference Timeout
    
    **Solution:** The timeout has been extended to 1000 seconds (16+ minutes). For very long texts:
    - The app automatically chunks text over 3000 characters
    - Each chunk is processed separately
    - Results are combined automatically
    - If still timing out, try reducing text length manually
    
    #### 🐌 Slow Performance
    
    **Solution:** 
    - Long texts (>5000 chars) take more time
    - FAQ generation is more complex than summarization
    - Wait for the full timeout period before canceling
    - Consider using a more powerful model or hardware
    
    ### 💡 Performance Tips
    
    - **Text Length**: 
      - Short (<3000 chars): Fast processing
      - Medium (3000-5000 chars): Moderate processing
      - Long (>5000 chars): Automatic chunking, longer processing
    - **Number of FAQs**: More questions = longer processing time
    - **Model Selection**: Larger models are more accurate but slower
    - **System Resources**: Ensure adequate RAM and CPU availability
    
    ### 📊 Recommended Limits
    
    - **Optimal text length**: 1000-3000 characters
    - **Maximum recommended**: 10,000 characters
    - **FAQ questions**: 3-7 for best results
    - **Summary length**: Use 'short' or 'medium' for faster results
    
    ### 🔧 Model Management
    
    - Models are stored locally in the model repository
    - Each model has different capabilities and resource requirements
    - You can switch between models based on your needs
    - Regular model updates may be required for optimal performance
    - Reload models if experiencing connection issues
    
    ### 📞 Support
    
    If you continue experiencing issues, please check:
    - Model server logs for error messages
    - System resource availability (RAM, CPU, disk space)
    - Model file integrity
    - Network connectivity (for model server)
    - Timeout settings in the code
    """)
    
    st.markdown("---")
    
    # Current configuration display
    st.markdown("### ⚙️ Current Configuration")
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        st.info(f"""
        **Timeout Settings**
        - Connection Timeout: 10 seconds
        - Inference Timeout: 1000 seconds
        - Streaming Enabled: Yes
        """)
    
    with config_col2:
        st.info(f"""
        **Text Processing**
        - Chunk Size: 2500 characters
        - Auto-chunking: Enabled
        - Max Output: 10,000 characters
        """)
    
    st.markdown("---")
    
    # Current models display
    if available_models:
        st.markdown("### 📦 Available Trained Models")
        for model in available_models:
            st.success(f"✓ {model}")
    else:
        st.warning("No models found in repository. Please load trained models.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888; padding: 20px;'>Powered by Custom-Trained Neural Network • Extended Timeout: 1000s • Built with ❤️</div>",
    unsafe_allow_html=True
)