import streamlit as st
import pandas as pd
import json
import os
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    torch = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    AutoModelForCausalLM = None
    TRANSFORMERS_AVAILABLE = False
import tempfile
import zipfile
from utils.data_processing import DataProcessor
from utils.model_handler import ModelHandler
from utils.training import Trainer
from utils.evaluation import Evaluator

# Initialize session state
if 'training_started' not in st.session_state:
    st.session_state.training_started = False
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False
if 'training_metrics' not in st.session_state:
    st.session_state.training_metrics = {'loss': [], 'accuracy': [], 'epochs': []}
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'fine_tuned_model' not in st.session_state:
    st.session_state.fine_tuned_model = None
if 'fine_tuned_tokenizer' not in st.session_state:
    st.session_state.fine_tuned_tokenizer = None

st.set_page_config(
    page_title="LLM Fine-Tuning Studio",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 LLM Fine-Tuning Studio")
st.markdown("Fine-tune language models with an intuitive interface and real-time monitoring")

# Check for transformers availability
if not TRANSFORMERS_AVAILABLE:
    st.error("""
    ⚠️ **Missing Dependencies**: The transformers library is not installed.
    
    To enable full functionality, please install the transformers library:
    ```bash
    pip install transformers
    ```
    
    You can still explore the interface and upload datasets, but model training will be limited.
    """)
    st.markdown("---")

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Choose Section:",
    ["Dataset Upload", "Model Configuration", "Training", "Model Testing", "Model Management"]
)

# Initialize processors
data_processor = DataProcessor()
model_handler = ModelHandler()
trainer = Trainer()
evaluator = Evaluator()

if section == "Dataset Upload":
    st.header("📁 Dataset Upload & Preview")
    
    uploaded_file = st.file_uploader(
        "Upload your training dataset",
        type=['csv', 'json'],
        help="Upload a CSV or JSON file containing your training data"
    )
    
    if uploaded_file is not None:
        try:
            # Process uploaded file
            df = data_processor.load_dataset(uploaded_file)
            st.session_state.dataset = df
            
            st.success(f"Dataset loaded successfully! Shape: {df.shape}")
            
            # Display dataset preview
            st.subheader("Dataset Preview")
            st.dataframe(df.head(10))
            
            # Dataset statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Samples", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            
            # Column mapping for different tasks
            st.subheader("Column Mapping")
            task_type = st.selectbox(
                "Select Task Type",
                ["Text Classification", "Text Generation"],
                help="Choose the type of fine-tuning task"
            )
            
            if task_type == "Text Classification":
                text_column = st.selectbox("Text Column", df.columns)
                label_column = st.selectbox("Label Column", df.columns)
                
                if text_column and label_column:
                    st.session_state.text_column = text_column
                    st.session_state.label_column = label_column
                    st.session_state.task_type = "classification"
                    
                    # Show label distribution
                    label_dist = df[label_column].value_counts()
                    fig = px.bar(
                        x=label_dist.index, 
                        y=label_dist.values,
                        title="Label Distribution"
                    )
                    st.plotly_chart(fig)
                    
            elif task_type == "Text Generation":
                text_column = st.selectbox("Text Column", df.columns)
                
                if text_column:
                    st.session_state.text_column = text_column
                    st.session_state.task_type = "generation"
                    
                    # Show text length distribution
                    text_lengths = df[text_column].str.len()
                    fig = px.histogram(
                        x=text_lengths,
                        title="Text Length Distribution",
                        nbins=30
                    )
                    st.plotly_chart(fig)
            
            # Data validation
            validation_results = data_processor.validate_dataset(df, st.session_state.get('task_type'))
            if validation_results['valid']:
                st.success("✅ Dataset validation passed!")
            else:
                st.error(f"❌ Dataset validation failed: {validation_results['message']}")
                
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")

elif section == "Model Configuration":
    st.header("⚙️ Model Configuration")
    
    if 'dataset' not in st.session_state:
        st.warning("Please upload a dataset first!")
    else:
        # Model selection
        st.subheader("Model Selection")
        
        if st.session_state.get('task_type') == 'classification':
            available_models = [
                "distilbert-base-uncased",
                "bert-base-uncased",
                "roberta-base",
                "albert-base-v2"
            ]
        else:
            available_models = [
                "gpt2",
                "distilgpt2",
                "microsoft/DialoGPT-small"
            ]
        
        selected_model = st.selectbox(
            "Choose Pre-trained Model",
            available_models,
            help="Select a pre-trained model to fine-tune"
        )
        
        st.session_state.selected_model = selected_model
        
        # Training parameters
        st.subheader("Training Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            learning_rate = st.number_input(
                "Learning Rate",
                min_value=1e-6,
                max_value=1e-2,
                value=2e-5,
                format="%.2e",
                help="Learning rate for the optimizer"
            )
            
            num_epochs = st.slider(
                "Number of Epochs",
                min_value=1,
                max_value=10,
                value=3,
                help="Number of training epochs"
            )
            
            batch_size = st.selectbox(
                "Batch Size",
                [8, 16, 32],
                index=1,
                help="Training batch size"
            )
        
        with col2:
            max_length = st.slider(
                "Max Sequence Length",
                min_value=64,
                max_value=512,
                value=128,
                help="Maximum sequence length for tokenization"
            )
            
            warmup_steps = st.number_input(
                "Warmup Steps",
                min_value=0,
                max_value=1000,
                value=100,
                help="Number of warmup steps for learning rate scheduler"
            )
            
            eval_steps = st.number_input(
                "Evaluation Steps",
                min_value=10,
                max_value=500,
                value=50,
                help="Evaluate model every N steps"
            )
        
        # Store parameters in session state
        st.session_state.training_params = {
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'max_length': max_length,
            'warmup_steps': warmup_steps,
            'eval_steps': eval_steps
        }
        
        # Model preview
        if st.button("Load Model Preview"):
            with st.spinner("Loading model information..."):
                try:
                    model_info = model_handler.get_model_info(selected_model)
                    st.json(model_info)
                except Exception as e:
                    st.error(f"Error loading model info: {str(e)}")

elif section == "Training":
    st.header("🚀 Model Training")
    
    if 'dataset' not in st.session_state or 'selected_model' not in st.session_state:
        st.warning("Please upload a dataset and configure model parameters first!")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Training Configuration Summary")
            config_df = pd.DataFrame([
                ["Model", st.session_state.selected_model],
                ["Task Type", st.session_state.task_type],
                ["Dataset Size", len(st.session_state.dataset)],
                ["Learning Rate", st.session_state.training_params['learning_rate']],
                ["Epochs", st.session_state.training_params['num_epochs']],
                ["Batch Size", st.session_state.training_params['batch_size']]
            ], columns=["Parameter", "Value"])
            st.table(config_df)
        
        with col2:
            if not st.session_state.training_started:
                if st.button("🚀 Start Training", type="primary"):
                    st.session_state.training_started = True
                    st.session_state.training_complete = False
                    st.session_state.training_metrics = {'loss': [], 'accuracy': [], 'epochs': []}
                    st.rerun()
            else:
                if st.button("⏹️ Stop Training"):
                    st.session_state.training_started = False
                    st.rerun()
        
        # Training progress
        if st.session_state.training_started and not st.session_state.training_complete:
            st.subheader("Training Progress")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            metrics_container = st.empty()
            
            # Simulate training process (in real implementation, this would be actual training)
            try:
                model, tokenizer = trainer.train_model(
                    st.session_state.dataset,
                    st.session_state.selected_model,
                    st.session_state.training_params,
                    st.session_state.text_column,
                    st.session_state.get('label_column'),
                    st.session_state.task_type,
                    progress_callback=lambda epoch, loss, acc: self._update_training_progress(
                        epoch, loss, acc, progress_bar, status_text, metrics_container
                    )
                )
                
                st.session_state.fine_tuned_model = model
                st.session_state.fine_tuned_tokenizer = tokenizer
                st.session_state.training_complete = True
                st.session_state.model_trained = True
                st.success("🎉 Training completed successfully!")
                
            except Exception as e:
                st.error(f"Training failed: {str(e)}")
                st.session_state.training_started = False
        
        # Display training metrics
        if st.session_state.training_metrics['loss']:
            st.subheader("Training Metrics")
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Training Loss', 'Training Accuracy'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            fig.add_trace(
                go.Scatter(
                    x=st.session_state.training_metrics['epochs'],
                    y=st.session_state.training_metrics['loss'],
                    mode='lines+markers',
                    name='Loss',
                    line=dict(color='red')
                ),
                row=1, col=1
            )
            
            if st.session_state.training_metrics['accuracy']:
                fig.add_trace(
                    go.Scatter(
                        x=st.session_state.training_metrics['epochs'],
                        y=st.session_state.training_metrics['accuracy'],
                        mode='lines+markers',
                        name='Accuracy',
                        line=dict(color='blue')
                    ),
                    row=1, col=2
                )
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

elif section == "Model Testing":
    st.header("🧪 Model Testing")
    
    if not st.session_state.model_trained:
        st.warning("Please train a model first!")
    else:
        st.subheader("Test Your Fine-tuned Model")
        
        if st.session_state.task_type == "classification":
            test_text = st.text_area(
                "Enter text to classify:",
                placeholder="Type your text here...",
                height=100
            )
            
            if st.button("Classify Text") and test_text:
                with st.spinner("Classifying..."):
                    try:
                        result = evaluator.predict_classification(
                            st.session_state.fine_tuned_model,
                            st.session_state.fine_tuned_tokenizer,
                            test_text
                        )
                        
                        st.subheader("Classification Result")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Predicted Label", result['label'])
                            st.metric("Confidence", f"{result['confidence']:.2%}")
                        
                        with col2:
                            # Show probability distribution
                            if 'probabilities' in result:
                                prob_df = pd.DataFrame(
                                    list(result['probabilities'].items()),
                                    columns=['Label', 'Probability']
                                )
                                fig = px.bar(prob_df, x='Label', y='Probability', 
                                           title="Prediction Probabilities")
                                st.plotly_chart(fig)
                        
                    except Exception as e:
                        st.error(f"Classification failed: {str(e)}")
        
        elif st.session_state.task_type == "generation":
            prompt_text = st.text_area(
                "Enter prompt for text generation:",
                placeholder="Start typing your prompt...",
                height=100
            )
            
            col1, col2 = st.columns(2)
            with col1:
                max_length = st.slider("Max Length", 10, 200, 50)
            with col2:
                temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
            
            if st.button("Generate Text") and prompt_text:
                with st.spinner("Generating..."):
                    try:
                        result = evaluator.generate_text(
                            st.session_state.fine_tuned_model,
                            st.session_state.fine_tuned_tokenizer,
                            prompt_text,
                            max_length=max_length,
                            temperature=temperature
                        )
                        
                        st.subheader("Generated Text")
                        st.write(result['generated_text'])
                        
                    except Exception as e:
                        st.error(f"Text generation failed: {str(e)}")
        
        # Batch testing
        st.subheader("Batch Testing")
        test_file = st.file_uploader(
            "Upload test dataset",
            type=['csv', 'json'],
            help="Upload a test dataset for batch evaluation"
        )
        
        if test_file and st.button("Run Batch Evaluation"):
            with st.spinner("Running batch evaluation..."):
                try:
                    test_df = data_processor.load_dataset(test_file)
                    results = evaluator.batch_evaluate(
                        st.session_state.fine_tuned_model,
                        st.session_state.fine_tuned_tokenizer,
                        test_df,
                        st.session_state.text_column,
                        st.session_state.get('label_column'),
                        st.session_state.task_type
                    )
                    
                    st.subheader("Batch Evaluation Results")
                    
                    if st.session_state.task_type == "classification":
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Accuracy", f"{results['accuracy']:.2%}")
                        with col2:
                            st.metric("Precision", f"{results['precision']:.2%}")
                        with col3:
                            st.metric("Recall", f"{results['recall']:.2%}")
                        
                        # Confusion matrix
                        if 'confusion_matrix' in results:
                            fig = px.imshow(
                                results['confusion_matrix'],
                                title="Confusion Matrix",
                                color_continuous_scale="Blues"
                            )
                            st.plotly_chart(fig)
                    
                except Exception as e:
                    st.error(f"Batch evaluation failed: {str(e)}")

elif section == "Model Management":
    st.header("💾 Model Management")
    
    if not st.session_state.model_trained:
        st.warning("No trained model available!")
    else:
        st.subheader("Save & Download Model")
        
        model_name = st.text_input(
            "Model Name",
            value=f"fine_tuned_{st.session_state.selected_model.replace('/', '_')}",
            help="Enter a name for your fine-tuned model"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save Model Locally"):
                with st.spinner("Saving model..."):
                    try:
                        save_path = f"./models/{model_name}"
                        os.makedirs(save_path, exist_ok=True)
                        
                        st.session_state.fine_tuned_model.save_pretrained(save_path)
                        st.session_state.fine_tuned_tokenizer.save_pretrained(save_path)
                        
                        st.success(f"Model saved to {save_path}")
                        
                    except Exception as e:
                        st.error(f"Failed to save model: {str(e)}")
        
        with col2:
            if st.button("📦 Create Download Package"):
                with st.spinner("Creating download package..."):
                    try:
                        # Create temporary directory
                        with tempfile.TemporaryDirectory() as temp_dir:
                            model_path = os.path.join(temp_dir, model_name)
                            os.makedirs(model_path)
                            
                            # Save model and tokenizer
                            st.session_state.fine_tuned_model.save_pretrained(model_path)
                            st.session_state.fine_tuned_tokenizer.save_pretrained(model_path)
                            
                            # Create metadata file
                            metadata = {
                                'model_name': model_name,
                                'base_model': st.session_state.selected_model,
                                'task_type': st.session_state.task_type,
                                'training_params': st.session_state.training_params,
                                'dataset_info': {
                                    'shape': st.session_state.dataset.shape,
                                    'columns': list(st.session_state.dataset.columns)
                                }
                            }
                            
                            with open(os.path.join(model_path, 'metadata.json'), 'w') as f:
                                json.dump(metadata, f, indent=2)
                            
                            # Create zip file
                            zip_path = os.path.join(temp_dir, f"{model_name}.zip")
                            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                                for root, dirs, files in os.walk(model_path):
                                    for file in files:
                                        file_path = os.path.join(root, file)
                                        arcname = os.path.relpath(file_path, temp_dir)
                                        zipf.write(file_path, arcname)
                            
                            # Provide download
                            with open(zip_path, 'rb') as f:
                                st.download_button(
                                    label="📥 Download Model Package",
                                    data=f.read(),
                                    file_name=f"{model_name}.zip",
                                    mime="application/zip"
                                )
                        
                    except Exception as e:
                        st.error(f"Failed to create download package: {str(e)}")
        
        # Model information
        st.subheader("Model Information")
        if hasattr(st.session_state.fine_tuned_model, 'config'):
            config_dict = st.session_state.fine_tuned_model.config.to_dict()
            st.json(config_dict)
        
        # Training summary
        if st.session_state.training_metrics['loss']:
            st.subheader("Training Summary")
            summary_data = {
                'Final Loss': st.session_state.training_metrics['loss'][-1] if st.session_state.training_metrics['loss'] else 'N/A',
                'Final Accuracy': st.session_state.training_metrics['accuracy'][-1] if st.session_state.training_metrics['accuracy'] else 'N/A',
                'Total Epochs': len(st.session_state.training_metrics['loss']),
                'Best Loss': min(st.session_state.training_metrics['loss']) if st.session_state.training_metrics['loss'] else 'N/A'
            }
            
            summary_df = pd.DataFrame(
                list(summary_data.items()),
                columns=['Metric', 'Value']
            )
            st.table(summary_df)

# Sidebar information
with st.sidebar:
    st.markdown("---")
    st.subheader("📊 System Status")
    
    # GPU/CPU status
    if TRANSFORMERS_AVAILABLE:
        device = "GPU" if torch.cuda.is_available() else "CPU"
        st.metric("Device", device)
        
        # Memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
            memory_total = torch.cuda.memory_reserved() / 1024**3
            st.metric("GPU Memory", f"{memory_used:.1f}GB / {memory_total:.1f}GB")
    else:
        st.metric("Device", "CPU (torch not available)")
    
    st.markdown("---")
    st.subheader("📚 Quick Help")
    
    with st.expander("Dataset Format"):
        st.markdown("""
        **Text Classification:**
        - CSV with 'text' and 'label' columns
        - JSON with text-label pairs
        
        **Text Generation:**
        - CSV with 'text' column
        - JSON with text examples
        """)
    
    with st.expander("Model Selection"):
        st.markdown("""
        **For Classification:**
        - BERT variants for high accuracy
        - DistilBERT for speed
        
        **For Generation:**
        - GPT-2 for general text
        - DialoGPT for conversations
        """)
    
    with st.expander("Training Tips"):
        st.markdown("""
        - Start with lower learning rates (2e-5)
        - Use smaller batch sizes for limited memory
        - Monitor validation loss to avoid overfitting
        - Increase epochs gradually
        """)

def _update_training_progress(self, epoch, loss, accuracy, progress_bar, status_text, metrics_container):
    """Update training progress in real-time"""
    progress = epoch / st.session_state.training_params['num_epochs']
    progress_bar.progress(progress)
    
    status_text.text(f"Epoch {epoch}/{st.session_state.training_params['num_epochs']} - Loss: {loss:.4f}")
    
    # Update metrics
    st.session_state.training_metrics['epochs'].append(epoch)
    st.session_state.training_metrics['loss'].append(loss)
    if accuracy is not None:
        st.session_state.training_metrics['accuracy'].append(accuracy)
    
    # Update metrics chart
    with metrics_container.container():
        if len(st.session_state.training_metrics['loss']) > 1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=st.session_state.training_metrics['epochs'],
                y=st.session_state.training_metrics['loss'],
                mode='lines+markers',
                name='Loss'
            ))
            if st.session_state.training_metrics['accuracy']:
                fig.add_trace(go.Scatter(
                    x=st.session_state.training_metrics['epochs'],
                    y=st.session_state.training_metrics['accuracy'],
                    mode='lines+markers',
                    name='Accuracy',
                    yaxis='y2'
                ))
            
            fig.update_layout(
                title="Training Progress",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                yaxis2=dict(title="Accuracy", overlaying='y', side='right'),
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
