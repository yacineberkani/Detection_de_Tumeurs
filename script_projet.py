import os
import pathlib
import numpy as np
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers

def charger_images(chemin):
    """Charge les images depuis le chemin donné et retourne les données et les étiquettes."""
    Tumeur_Non = [f for f in os.listdir(chemin / 'non') if f.endswith('.jpg')]
    Tumeur_Oui = [f for f in os.listdir(chemin / 'oui') if f.endswith('.jpg')]
    test_Non = [f for f in os.listdir(chemin / 'non_test') if f.endswith('.jpg')]
    test_Oui = [f for f in os.listdir(chemin / 'oui_test') if f.endswith('.jpg')]

    data, label, data_test, label_test  = [], [], [], [] 

    for nom_image in Tumeur_Non:
        image = cv2.imread(str(chemin / 'non' / nom_image))
        image = Image.fromarray(image, 'RGB').resize((128, 128))
        data.append(np.array(image))
        label.append(0)

    for nom_image in Tumeur_Oui:
        image = cv2.imread(str(chemin / 'oui' / nom_image))
        image = Image.fromarray(image, 'RGB').resize((128, 128))
        data.append(np.array(image))
        label.append(1)

    for nom_image in test_Non:
        image = cv2.imread(str(chemin / 'non_test' / nom_image))
        image = Image.fromarray(image, 'RGB').resize((128, 128))
        data_test.append(np.array(image))
        label_test.append(0)

    for nom_image in test_Oui:
        image = cv2.imread(str(chemin / 'oui_test' / nom_image))
        image = Image.fromarray(image, 'RGB').resize((128, 128))
        data_test.append(np.array(image))
        label_test.append(1)
    

    return np.array(data), np.array(label), np.array(data_test), np.array(label_test)

def entrainer_random_forest(x_train, y_train, x_val, y_val,x_test,y_test):
    """Entraîne un modèle Random Forest et évalue sa précision."""
    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(x_train.reshape(x_train.shape[0], -1), y_train)
    accuracy = accuracy_score(y_val, clf.predict(x_val.reshape(x_val.shape[0], -1)))
    accuracy_test = accuracy_score(y_test, clf.predict(x_test.reshape(x_test.shape[0], -1)))
    y_pred = clf.predict(x_test.reshape(x_test.shape[0], -1))
    cmrf = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(cmrf)
    display.plot()
    plt.show()
    print(f"Accuracy (Random Forest): {accuracy}")
    print(f"Accuracy (Random Forest) sur les donnée de test : {accuracy_test}")
    return clf

def construire_evaluer_modele_CNN(x_train, y_train, x_val, y_val, x_test, y_test):
    """Construit, entraîne et évalue un modèle CNN."""
    cnn = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    cnn.summary()
    cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    cnnhistory = cnn.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_val, y_val))
    cnn.evaluate(x_test, y_test)
    y_pred = cnn.predict(x_test)
    y_pred = (y_pred > 0.5).astype(int)
    print('Classification Report\n', classification_report(y_test, y_pred))
    cmcnn = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(cmcnn)
    display.plot()
    plt.show()
    return cnn, cnnhistory

def construire_evaluer_modele_VGG16(x_train, y_train, x_val, y_val, x_test, y_test):
    """Construit, entraîne et évalue un modèle VGG16."""
    base_model = VGG16(include_top=False, weights="imagenet", input_shape=(128, 128, 3))
    for layer in base_model.layers:
        layer.trainable = False

    model_VGG16 = Sequential([base_model, Flatten(), Dense(64, activation='relu'), Dense(1, activation='sigmoid')])
    model_VGG16.summary()
    model_VGG16.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    vgghistory = model_VGG16.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_val, y_val))
    model_VGG16.evaluate(x_test, y_test)
    y_pred = model_VGG16.predict(x_test)
    y_pred = (y_pred > 0.5).astype(int)
    print('Classification Report\n', classification_report(y_test, y_pred))
    cmvgg = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(cmvgg)
    display.plot()
    plt.show()
    return model_VGG16, vgghistory

def afficher_resultats(history):
    """Affiche les graphiques de performance du modèle."""
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], 'bo-', label="Perte d'entraînement")
    plt.plot(history.history['val_loss'], 'ro-', label="Perte de validation")
    plt.title("Perte d'entraînement et de validation")
    plt.xlabel("Époques")
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], 'bo-', label="Précision d'entraînement")
    plt.plot(history.history['val_accuracy'], 'ro-', label="Précision de validation")
    plt.title("Précision d'entraînement et de validation")
    plt.xlabel('Époques')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
def prediction_generique(modele, chemin_image, type_modele='cnn'):
    img = cv2.imread(chemin_image)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (128, 128))
    
    if type_modele == 'clf':
        # Pour Random Forest: Aplatir et normaliser l'image
        img_flattened = img_resized.flatten().reshape(1, -1) / 255.0
        prediction = modele.predict(img_flattened)
        prediction = prediction[0]
    else:
        # Pour CNN et VGG16: Étendre les dimensions et normaliser
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = modele.predict(img_array)
        prediction = prediction[0][0]

    plt.imshow(img_rgb)
    plt.show()
    if prediction > 0.5:
        print(f"{chemin_image}: Tumeur cérébrale détectée")
    else:
        print(f"{chemin_image}: Tumeur cérébrale non détectée")