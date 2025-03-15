import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage import io
import matplotlib.pyplot as plt
from PIL import Image

# Fonction pour comparer les images avec SSIM
def compare_ssim(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    score, diff = ssim(img1_gray, img2_gray, full=True)
    
    fig, ax = plt.subplots()
    ax.imshow(diff, cmap='gray')
    ax.set_title("Carte des différences SSIM")
    
    return score, fig

# Fonction pour comparer les images avec ORB
def compare_orb(img1, img2):
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Correspondances ORB ({len(matches)} matches)")
    
    return len(matches), fig

# Fonction pour ajouter du texte à une image
def add_text_to_image(image, text, position, font_scale, color, thickness):
    font = cv2.FONT_HERSHEY_SIMPLEX
    image_with_text = cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return image_with_text

# Fonction pour appliquer des filtres
def apply_filter(image, filter_type):
    if filter_type == "Flou gaussien":
        return cv2.GaussianBlur(image, (15, 15), 0)
    elif filter_type == "Détection de contours (Canny)":
        return cv2.Canny(image, 100, 200)
    elif filter_type == "Seuillage (Thresholding)":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return binary_image
    else:
        return image

# Fonction pour afficher l'histogramme des couleurs
def plot_histogram(image):
    plt.figure()
    colors = ('b', 'g', 'r')
    for i, color in enumerate(colors):
        histogram = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(histogram, color=color)
    plt.title("Histogramme des couleurs")
    plt.xlabel("Intensité des pixels")
    plt.ylabel("Nombre de pixels")
    return plt

# Interface Streamlit
st.title("🔍 Application de Traitement d'Images")

st.sidebar.header("📂 Importer une image")
uploaded_file = st.sidebar.file_uploader("Choisir une image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Charger l'image
    image = np.array(Image.open(uploaded_file))
    st.subheader("Image chargée")
    st.image(image, caption="Image originale", use_column_width=True)

    # Menu pour choisir la fonctionnalité
    st.sidebar.header("🎛️ Fonctionnalités")
    feature = st.sidebar.selectbox(
        "Choisir une fonctionnalité",
        ("Ajouter du texte", "Appliquer un filtre", "Analyser l'histogramme")
    )

    if feature == "Ajouter du texte":
        st.subheader("Ajouter du texte à l'image")
        text = st.text_input("Entrez le texte à ajouter", "Hello World")
        position_x = st.slider("Position X", 0, image.shape[1], 50)
        position_y = st.slider("Position Y", 0, image.shape[0], 50)
        font_scale = st.slider("Taille de la police", 0.5, 5.0, 1.0)
        color = st.color_picker("Couleur du texte", "#FFFFFF")
        thickness = st.slider("Épaisseur du texte", 1, 10, 2)
        
        # Convertir la couleur hexadécimale en BGR
        color_bgr = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))
        
        image_with_text = add_text_to_image(image, text, (position_x, position_y), font_scale, color_bgr, thickness)
        st.image(image_with_text, caption="Image avec texte", use_column_width=True)

    elif feature == "Appliquer un filtre":
        st.subheader("Appliquer un filtre à l'image")
        filter_type = st.selectbox(
            "Choisir un filtre",
            ("Flou gaussien", "Détection de contours (Canny)", "Seuillage (Thresholding)")
        )
        filtered_image = apply_filter(image, filter_type)
        st.image(filtered_image, caption=f"Filtre appliqué : {filter_type}", use_column_width=True)

    elif feature == "Analyser l'histogramme":
        st.subheader("Analyse de l'histogramme des couleurs")
        histogram_plot = plot_histogram(image)
        st.pyplot(histogram_plot)

else:
    st.info("Veuillez charger une image pour commencer.")
