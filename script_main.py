import subprocess 
from script_dataaug import *
from script_projet import *

def main():
    # Installe les dépendances à partir du fichier requirements.txt
    command = "pip install -r requirements.txt"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode == 0:
        print(f"Requirements installed successfully:\n{stdout.decode('utf-8')}")
    else:
        print(f"Error installing requirements:\n{stderr.decode('utf-8')}")

    """Fonction principale pour l'augmentation des images."""
    # Compter les images avant augmentation
    compter_images('../IRM', 'non')
    compter_images('../test', 'non_test')
    compter_images('../IRM', 'oui')
    compter_images('../test', 'oui_test')

    # Augmenter les images
    augmenter_images('../IRM/non', '../data_aug/non')
    augmenter_images('../test/non_test', '../data_aug/non_test',nb_augmentations=2)
    augmenter_images('../IRM/oui', '../data_aug/oui')
    augmenter_images('../test/oui_test', '../data_aug/oui_test',nb_augmentations=2)

    # Compter les images après augmentation
    compter_images('../data_aug', 'non')
    compter_images('../data_aug', 'non_test')
    compter_images('../data_aug', 'oui')
    compter_images('../data_aug', 'oui_test')


    """Fonction principale pour le traitement des images."""
    # Charger les images
    data, label, data_test, label_test = charger_images(pathlib.Path('../data_aug'))

    # Division des ensembles d'entraînement et de validation
    x_train, x_val, y_train, y_val = train_test_split(data, label, test_size=0.1, random_state=42)
    x_test, y_test = data_test, label_test
    x_train, x_val, x_test  = normalize(x_train, axis=1), normalize(x_val, axis=1), normalize(x_test, axis=1)


    # Entraînement du modèle Random Forest
    Clf = entrainer_random_forest(x_train, y_train, x_val, y_val,x_test,y_test)
    Clf

    # Construction et évaluation du modèle CNN
    CNN, cnnhistory= construire_evaluer_modele_CNN(x_train, y_train, x_val, y_val, x_test, y_test)
    # Construction et évaluation du modèle VGG16
    model_VGG16, vgghistory = construire_evaluer_modele_VGG16(x_train, y_train, x_val, y_val, x_test, y_test)

    # Affichage des résultats
    afficher_resultats(cnnhistory)
    afficher_resultats(vgghistory)

    # Sauvegarde du modèle VGG16
    CNN.save('../model_CNN.h5')
    model_VGG16.save('../model_VGG16.h5')
    


    # Charger les images de test final
    dossier_images = '../test_final'
    for fichier in os.listdir(dossier_images):
        chemin_complet = os.path.join(dossier_images, fichier)
        if not fichier.startswith('.') and os.path.isfile(chemin_complet):
            print('#################( Clf )#####################')
            print(chemin_complet)
            print('#################( Clf )#####################')
            prediction_generique(Clf, chemin_complet, type_modele='clf')   
    

    for fichier in os.listdir(dossier_images):
        chemin_complet = os.path.join(dossier_images, fichier)
        if not fichier.startswith('.') and os.path.isfile(chemin_complet):
            print('#################( CNN )#####################')
            print(chemin_complet)
            print('#################( CNN )#####################')
            prediction_generique(CNN, chemin_complet, type_modele='CNN')
   
    for fichier in os.listdir(dossier_images):
        chemin_complet = os.path.join(dossier_images, fichier)
        if not fichier.startswith('.') and os.path.isfile(chemin_complet):
            print('###################( VGG16 )###################')
            print(chemin_complet)
            print('###################( VGG16 )###################')
            prediction_generique(model_VGG16, chemin_complet, type_modele='cnn')
if __name__ == "__main__":
    main()
