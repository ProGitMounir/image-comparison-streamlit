# 🔍 Application de Traitement et Comparaison d'Images

Une application Streamlit pour traiter et comparer des images en utilisant différentes techniques de vision par ordinateur, telles que l'ajout de texte, l'application de filtres, l'analyse d'histogramme, et la comparaison d'images avec SSIM et ORB.

---

## 📋 Table des matières
- [Fonctionnalités](#-fonctionnalités)
- [Technologies utilisées](#-technologies-utilisées)
- [Installation](#-installation)

---

## 🚀 Fonctionnalités

- **Ajout de texte** : Ajoutez du texte à une image en choisissant la position, la taille, la couleur et l'épaisseur du texte.
- **Application de filtres** : Appliquez des filtres tels que le flou gaussien, la détection de contours (Canny), et le seuillage (thresholding).
- **Analyse d'histogramme** : Visualisez la distribution des intensités de pixels pour chaque canal de couleur (rouge, vert, bleu).
- **Comparaison d'images** :
  - **SSIM (Structural Similarity Index)** : Comparez deux images en mesurant leur similarité structurelle.
  - **ORB (Oriented FAST and Rotated BRIEF)** : Détectez des correspondances entre les images en utilisant des points clés et des descripteurs.

---

## 🛠️ Technologies utilisées

- **Streamlit** : Pour créer l'interface utilisateur interactive.
- **OpenCV (cv2)** : Pour le traitement d'images (filtres, ajout de texte, etc.).
- **Scikit-image (skimage)** : Pour la comparaison SSIM.
- **Matplotlib** : Pour la visualisation des histogrammes et des différences SSIM.
- **NumPy** : Pour la manipulation des tableaux d'images.

---

## 📥 Installation

1. Clonez ce dépôt :
   ```
   git clone https://github.com/votre-utilisateur/votre-repo.git
   cd votre-repo
   
2. Installez les dépendances :
    ```
    pip install -r requirements.txt

3. Lancez l'application Streamlit :
    ```
    streamlit run app.py
    
Ouvrez votre navigateur et accédez à l'application

👤 **MOUNIR IYA AMINE**  
📧 Contact : `pro.mailmounir@gmail.com`  
🌍 GitHub : ProGitMounir(https://github.com/ProGitMounir)

