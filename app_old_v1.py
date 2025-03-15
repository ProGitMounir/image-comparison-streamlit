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

# Interface Streamlit
st.title("🔍 Comparaison d'Images avec SSIM et ORB")

st.sidebar.header("📂 Importer deux images")
uploaded_file1 = st.sidebar.file_uploader("Choisir la première image", type=["jpg", "png", "jpeg"])
uploaded_file2 = st.sidebar.file_uploader("Choisir la deuxième image", type=["jpg", "png", "jpeg"])

if uploaded_file1 and uploaded_file2:
    # Charger les images
    image1 = np.array(Image.open(uploaded_file1))
    image2 = np.array(Image.open(uploaded_file2))

    # Affichage des images
    st.subheader("Images chargées")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image1, caption="Image 1", use_column_width=True)
    with col2:
        st.image(image2, caption="Image 2", use_column_width=True)

    # Sélection de la méthode
    method = st.sidebar.radio("Méthode de comparaison", ("SSIM", "ORB"))

    # Comparaison
    if method == "SSIM":
        st.subheader("Résultat de la comparaison SSIM")
        st.markdown("""
        **Qu'est-ce que SSIM ?**  
        L'indice de similarité structurelle (SSIM) est une méthode qui compare deux images en mesurant leur similarité en termes de luminance, contraste et structure.  
        - **Score SSIM** : Un score proche de 1 indique une forte similarité, tandis qu'un score proche de 0 indique une faible similarité.  
        - **Carte des différences** : La carte montre les zones où les images diffèrent (plus claires = différences plus importantes).
        """)
        score, fig = compare_ssim(image1, image2)
        st.write(f"**Indice de similarité SSIM :** {score:.4f}")
        st.pyplot(fig)
    else:
        st.subheader("Résultat de la comparaison ORB")
        st.markdown("""
        **Qu'est-ce que ORB ?**  
        ORB (Oriented FAST and Rotated BRIEF) est un algorithme de détection de points clés et de descripteurs. Il est utilisé pour trouver des correspondances entre les images.  
        - **Nombre de correspondances** : Plus le nombre de correspondances est élevé, plus les images sont similaires.  
        - **Visualisation** : Les lignes relient les points clés correspondants entre les deux images.
        """)
        num_matches, fig = compare_orb(image1, image2)
        st.write(f"**Nombre de correspondances détectées :** {num_matches}")
        st.pyplot(fig)

else:
    st.info("Veuillez charger deux images pour commencer la comparaison.")