import os
import pathlib
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

def compter_images(chemin, dossier):
    """Compte le nombre d'images dans le dossier spécifié."""
    chemin_complet = pathlib.Path(chemin) / dossier
    images = os.listdir(str(chemin_complet))
    print(f"Le nombre d'images dans {dossier} est {len(images)}")
    return images

def augmenter_images(folder_path, save_to_dir, nb_augmentations=5):
    """Effectue l'augmentation d'images et les sauvegarde."""
    datagen = ImageDataGenerator(
        rotation_range=0,
        width_shift_range=0.05,
        height_shift_range=0.05,
        rescale=1./255,
        shear_range=0.05,
        brightness_range=[0.1, 1.5],
        horizontal_flip=True,
        vertical_flip=True
    )

    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, filename)
            img = load_img(img_path)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            for i, _ in enumerate(datagen.flow(x, batch_size=1, save_to_dir=save_to_dir, 
                                               save_prefix=filename.split('.')[0], 
                                               save_format='jpg'), 1):
                if i > nb_augmentations:
                    break