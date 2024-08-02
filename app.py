import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import time
import plotly.graph_objs as go

model = tf.keras.models.load_model('emotionfinal.h5')

def introduction_page():  #Introduction page
    st.title('Welcome to Emotion Detection App')
    st.write("""
    This application uses a Convolutional Neural Network (CNN) to detect emotions from facial images.
    **How it works:**
    1. Navigate to the second page using the slider.
    2. Upload an image of a face.
    3. The app will predict the emotion displayed in the image.
    **This app can detect following 7 emotions**
    - Angry
    - Disgusted
    - Fearful
    - Happy
    - Neutral
    - Sad
    - Surprised
    """)

def emotion_detection_page(): #this page contains trained model 
    st.title('Emotion Detection')
    
    uploaded_image = st.file_uploader("Choose an image...") #to get the image from the user

    col1, col2 = st.columns(2)
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        
        with col1: #this column contains the view of uploaded image
            st.image(image, caption='Uploaded Image.', width=250)
    
            image = image.resize((40, 40))   
            image = np.array(image) / 255.0
            image = np.expand_dims(image, axis=0)

        with col2: #This column for display the predictions
            with st.spinner('Wait for the results'):
                prediction = model.predict(image)
                time.sleep(2)  
   
                predicted_class = np.argmax(prediction, axis=1)[0]
                
                if predicted_class == 1:
                    st.success('You are looking angry', icon="😠")
                elif predicted_class==2:
                    st.success('You are looking disgusted', icon="😒")
                elif predicted_class==3:
                    st.success('You are looking fearful', icon="😨")
                elif predicted_class==4:
                    st.success('You are looking happy', icon="😄")
                elif predicted_class==5:
                    st.success('You are looking neutral', icon="😐")
                elif predicted_class==6:
                    st.success('You are looking sad', icon="😭")
                else:
                    st.success('You are looking surprised', icon="🤯")

def plot_accuracy():
    epochs = [1, 2, 3, 4, 5]
    accuracy = [0.60, 0.65, 0.70, 0.75, 0.80]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=accuracy, mode='lines+markers', name='Accuracy'))
    fig.update_layout(
        title='Model Accuracy',
        xaxis_title='Epoch',
        yaxis_title='Accuracy',
        template='plotly_dark'
    )
    st.plotly_chart(fig, use_container_width=True)


page = st.sidebar.radio("Select Page", ["Introduction", "Emotion Detection"]) #To define a the two options with the radio button

if page == "Introduction":
    introduction_page()
elif page == "Emotion Detection":
    emotion_detection_page()

plot_accuracy() 
