import os
import sys
import glob
import cv2
import numpy as np
import _pickle as cPickle
from tqdm import tqdm
sys.path.append('../lib')
from align import align_nocs_to_depth
from utils import load_depth


def create_img_list(data_dir):
    """ Create train/val/test data list for CAMERA and Real. """
    # # CAMERA dataset
    # for subset in ['train', 'val']:
    #     img_list = []
    #     img_dir = os.path.join(data_dir, 'CAMERA', subset)
    #     folder_list = [name for name in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, name))]
    #     for i in range(10*len(folder_list)):
    #         folder_id = int(i) // 10
    #         img_id = int(i) % 10
    #         img_path = os.path.join(subset, '{:05d}'.format(folder_id), '{:04d}'.format(img_id))
    #         img_list.append(img_path)
    #     with open(os.path.join(data_dir, 'CAMERA', subset+'_list_all.txt'), 'w') as f:
    #         for img_path in img_list:
    #             f.write("%s\n" % img_path)
    # Real dataset
    for subset in ['train', 'test']:
        img_list = []
        img_dir = os.path.join(data_dir, 'Real', subset)
        folder_list = [name for name in sorted(os.listdir(img_dir)) if os.path.isdir(os.path.join(img_dir, name))]
        for folder in folder_list:
            img_paths = glob.glob(os.path.join(img_dir, folder, '*_color.png'))
            img_paths = sorted(img_paths)
            for img_full_path in img_paths:
                img_name = os.path.basename(img_full_path)
                img_ind = img_name.split('_')[0]
                img_path = os.path.join(subset, folder, img_ind)
                img_list.append(img_path)
        with open(os.path.join(data_dir, 'Real', subset+'_list_all.txt'), 'w') as f:
            for img_path in img_list:
                f.write("%s\n" % img_path)
    print('Write all data paths to file done!')


def process_data(img_path, depth):
    """ Load instance masks for the objects in the image.
    Inputs : 
        - img_path : string, préfixe d'un path de type data/Real/test/scene_6/0064 ou data/CAMERA/val/02445/0008 (pour une image données, tous ses fichiers .png color, coord
                    depth, mask et meta.txt ont comme préfixe cette string)
        - depth : array numpy 2D, carte de profondeur de l'image dont le préfixe de path est img_path
    
    Outputs : 
        - masks : array numpy 3D de shape (h,w,n) où h,w dimensions de l'image, n nombre d'objets dans l'image. Contient les masques binaires (booléens) de chaque objet 
                    dans l'image
        - coords : array numpy 4D (h,w,n,3) où h,w dimensions de l'image, n nombre d'objets dans l'image. Contient les maps NOCS de chaque objet dans l'image (valeurs entre
                    0 et 1)
        - class_ids :  liste d'ints qui contient les ids des catégories des objets dans l'image
        - instance_ids : liste d'ints qui contient les numéros des objets dans l'image
        - model_list :  liste de strings qui contient les noms des modèles des objets dans l'image
        - bboxes :  array numpy 2D (n,4) où n nombre d'objets dans l'image. Contient les bboxes (y1, x1, y2, x2) de chaque objet dans l'image.
    """
    mask_path = img_path + '_mask.png'  #path du masque binaire des objets de l'image
    mask = cv2.imread(mask_path)[:, :, 2]  #array numpy 2D, contient le masque binaire de chaque objet dans l'image. Pixels blancs = 255, pixels appartenant à l'objet de
    # d'id de catégorie n (entre 1 et 6) = n
    mask = np.array(mask, dtype=np.int32)  #transforme le masque en array de ints
    all_inst_ids = sorted(list(np.unique(mask)))  #liste des ids des objets présents dans la scène + le pixel blanc 255 (les objets sont numérotés dans la scène)
    assert all_inst_ids[-1] == 255  #vérifie que le pixel blanc est bien dans le masque
    del all_inst_ids[-1]  #enlève le pixel blanc de la liste des ids des objets dans la scène
    num_all_inst = len(all_inst_ids)  #nombre d'objets présents dans la scène 
    h, w = mask.shape  #hauteur et largeur de l'image

    coord_path = img_path + '_coord.png'  #path de la map Normalized Object Coordinate Space de l'image
    coord_map = cv2.imread(coord_path)[:, :, :3]  #array numpy 3D, contient la map Normalized Object Coordinate Space de l'image
    coord_map = coord_map[:, :, (2, 1, 0)]  #réagence les 3 colonnes de la dimension 3, car pour cv2 le format est BGR et pas RGB
    # flip z axis of coord map
    coord_map = np.array(coord_map, dtype=np.float32) / 255  #met l'array en floats et remet les coordonnées entre 0 et 1
    coord_map[:, :, 2] = 1 - coord_map[:, :, 2]  #?

    class_ids = []  #liste des ids des catégories des objets présents dans le fichier meta.txt
    instance_ids = []  #liste des ids des instances (numérotation) présents dans le fichier meta.txt de l'image
    model_list = []  #liste des noms des objets présents dans le fichier meta.txt de l'image
    masks = np.zeros([h, w, num_all_inst], dtype=np.uint8)  #array 3D de shape (h, w, num_all_inst), pour chaque objet dans l'image contient son masque binaire (array 2D de
    #booléens)
    coords = np.zeros((h, w, num_all_inst, 3), dtype=np.float32)  #array 4D de shape (h, w, num_all_inst,3), pour chaque objet dans l'image contient sa map NOCS
    bboxes = np.zeros((num_all_inst, 4), dtype=np.int32)  #array 2D de shape (num_all_inst,4), pour chaque objet dans l'image contient les coordonnées de sa bbox (indices
    #dans l'array de l'image)

    meta_path = img_path + '_meta.txt'  #path du fichier qui contient la description de la scène de l'image
    with open(meta_path, 'r') as f:  #ouvre le fichier
        i = 0
        for line in f:  #parcourt les lignes du fichier, ie les objets dans l'image
            line_info = line.strip().split(' ')  #liste des mots sur la ligne
            inst_id = int(line_info[0])  #id de l'objet dans la scène (son numéro )
            cls_id = int(line_info[1])  #id de la catégorie de l'objet

            if cls_id == 0 or (inst_id not in all_inst_ids):  #si l'objet est un objet n'appartenant à aucune des catégories prédéfinies ou n'est pas dans l'image
                continue  #passe à l'objet suivant

            if len(line_info) == 3:  #cas où l'objet est un objet réel (issu d'une image du test set de Real)
                model_id = line_info[2]    #nom de l'objet, définit son id
            else:   #cas où l'objet est un objet synthétique (issu d'une image du test set de CAMERA)
                model_id = line_info[3]    #id de l'objet (hash)
            
            # remove one mug instance in CAMERA train due to improper model
            if model_id in ['b9be7cfe653740eb7633a2dd89cec754', 'd3b53f56b4a7b3b3c9f016d57db96408'] :  #modifié, enlève les objets dont le modèle est manquant/mauvais
            #(dans CAMERA)
                continue  #passe à l'objet suivant

            
            inst_mask = np.equal(mask, inst_id)  #array numpy 2D de la même shape que mask rempli de booléen qui indiquent les pixels qui appartiennent à l'objet
            
            #crée la bounding box
            horizontal_indicies = np.where(np.any(inst_mask, axis=0))[0]  #array 1D, indices des lignes de mask où les pixels appartiennent à l'objet en cours
            vertical_indicies = np.where(np.any(inst_mask, axis=1))[0]  #array 1D, indices des colonnes de mask où les pixels appartiennent à l'objet en cours
            assert horizontal_indicies.shape[0], print(img_path)
            x1, x2 = horizontal_indicies[[0, -1]]  #abscisses limites de la bounding box (indices de mask)
            y1, y2 = vertical_indicies[[0, -1]]  #ordonnées limites de la bounding box (indices de mask)
            
            # x2 et y2 ne doivent pas faire partie de la box, on incrémente de 1.
            x2 += 1
            y2 += 1
            
            if np.any(np.logical_or((x2-x1) > 600, (y2-y1) > 440)):  #cas où l'objet occupe toute l'image, erreur de rendering qui arrive dans CAMERA
                return None, None, None, None, None, None  #ne retourne rien
            
            
            final_mask = np.logical_and(inst_mask, depth > 0)  #array 2D de booléens de la même shape que mask, contient le masque de l'objet en cours en ne conservant que
            #les pixels pour lesquels la profondeur n'est pas nulle 

            if np.sum(final_mask) < 64:  #si le masque est composé de moins de 64 pixels
                continue  #ignore l'objet, pas assez de pixels de depth
            class_ids.append(cls_id)  #ajoute le numéro de la catégorie de l'objet à la liste des catégories des objets présents dans l'image
            instance_ids.append(inst_id)  #ajoute le numéro de l'objet dans l'image à la liste des des ids des objets présents dans l'image
            model_list.append(model_id)  #ajoute le nom du modèle de l'objet à la liste des noms des modèles des objets présents dans l'image
            masks[:, :, i] = inst_mask  #ajoute le masque binaire de l'objet dans l'image
            coords[:, :, i, :] = np.multiply(coord_map, np.expand_dims(inst_mask, axis=-1))  #ajoute la map NOCS de l'objet dans l'image
            bboxes[i] = np.array([y1, x1, y2, x2])  #ajoute la bbox de l'objet dans l'image
            i += 1  #incrémente à chaque fois qu'un objet est valide

    if i == 0:  #si aucun objet n'a été correctement détecté depuis meta.txt
        return None, None, None, None, None, None  #ne retourne rien

    masks = masks[:, :, :i]  #ne conserve que les masques binaires des objets qui ont été correctement détectés depuis meta.txt
    coords = np.clip(coords[:, :, :i, :], 0, 1)  #ne conserve que les maps NOCS des objets qui ont été correctement détectés depuis meta.txt (valeurs entre 0 et 1)
    bboxes = bboxes[:i, :]  #ne conserve que les maps NOCS des objets qui ont été correctement détectés depuis meta.txt

    return masks, coords, class_ids, instance_ids, model_list, bboxes


def annotate_camera_train(data_dir):
    """ Generate gt labels for CAMERA train data. """
    camera_train = open(os.path.join(data_dir, 'CAMERA', 'train_list_all.txt')).read().splitlines()
    intrinsics = np.array([[577.5, 0, 319.5], [0, 577.5, 239.5], [0, 0, 1]])
    # meta info for re-label mug category
    with open(os.path.join(data_dir, 'obj_models/mug_meta.pkl'), 'rb') as f:
        mug_meta = cPickle.load(f)

    valid_img_list = []
    for img_path in tqdm(camera_train):
        img_full_path = os.path.join(data_dir, 'CAMERA', img_path)
        all_exist = os.path.exists(img_full_path + '_color.png') and \
                    os.path.exists(img_full_path + '_coord.png') and \
                    os.path.exists(img_full_path + '_depth.png') and \
                    os.path.exists(img_full_path + '_mask.png') and \
                    os.path.exists(img_full_path + '_meta.txt')
        if not all_exist:
            continue
        depth = load_depth(img_full_path)
        masks, coords, class_ids, instance_ids, model_list, bboxes = process_data(img_full_path, depth)
        if instance_ids is None:
            continue
        # Umeyama alignment of GT NOCS map with depth image
        scales, rotations, translations, error_messages, _ = \
            align_nocs_to_depth(masks, coords, depth, intrinsics, instance_ids, img_path)
        if error_messages:
            continue
        # re-label for mug category
        for i in range(len(class_ids)):
            if class_ids[i] == 6:
                T0 = mug_meta[model_list[i]][0]
                s0 = mug_meta[model_list[i]][1]
                T = translations[i] - scales[i] * rotations[i] @ T0
                s = scales[i] / s0
                scales[i] = s
                translations[i] = T
        # write results
        gts = {}
        gts['class_ids'] = class_ids    # int list, 1 to 6
        gts['bboxes'] = bboxes  # np.array, [[y1, x1, y2, x2], ...]
        gts['scales'] = scales.astype(np.float32)  # np.array, scale factor from NOCS model to depth observation
        gts['rotations'] = rotations.astype(np.float32)    # np.array, R
        gts['translations'] = translations.astype(np.float32)  # np.array, T
        gts['instance_ids'] = instance_ids  # int list, start from 1
        gts['model_list'] = model_list  # str list, model id/name
        with open(img_full_path + '_label.pkl', 'wb') as f:
            cPickle.dump(gts, f)
        valid_img_list.append(img_path)
    # write valid img list to file
    with open(os.path.join(data_dir, 'CAMERA/train_list.txt'), 'w') as f:
        for img_path in valid_img_list:
            f.write("%s\n" % img_path)


def annotate_real_train(data_dir):
    """ Generate gt labels for Real train data through PnP.
        Enregistre dans Real/train pour chaque image de chaque scène un fichier '****_label.pkl' qui contient un dictionnaire de clefs/valeurs :
            - 'class_ids' : array numpy 1D d'ints qui contient la liste des ids des catégories des objets dans l'image
            - 'instance_ids' : array numpy 1D d'ints qui contient la liste des numéros des objets dans l'image
            - 'model_list' : liste de strings qui contient les noms des modèles des objets dans l'image
            - 'size' : array numpy 2D (n,3) où n nombre d'instances dans l'image, contient pour chaque objet dans l'image ses dimensions 3D (ou celles de son modèle NOCS?)
            - 'scales' : array numpy 1D avec pour chaque objet dans l'image son scale factor from NOCS model to depth observation
            - 'rotations' : array numpy 3D (n,3,3), contient pour chaque objet dans l'image sa matrice de rotation
            - 'translations' : array numpy 2D (n,3), contient pour chaque objet dans l'image sa translation
            - 'poses' : array numpy 3D (n,3,3), contient pour chaque objet dans l'image sa rotation, la translation + 1 ligne (cf gt_Rts)
            - 'bboxes' : array numpy 2D (n,4), contient la bbox de chaque objet dans l'image (y1, x1, y2, x2)            
    """
    real_train = open(os.path.join(data_dir, 'Real/train_list_all.txt')).read().splitlines()  #liste de préfixes de paths des images dans le train set ('train/scene_*/****')
    intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])  #paramètres intrinsèques de la caméra pour le train set
    # scale factors for all instances
    scale_factors = {}  #dict avec en clefs les noms (sans les extensions, type bottle3_scene5_norm) des fichiers .txt des modèles d'objets du train set en clefs, et en 
    #valeurs leur facteur d'échelle
    path_to_size = glob.glob(os.path.join(data_dir, 'obj_models/real_train', '*_norm.txt'))  #liste de tous les paths relatifs à 'data/obj_models/real_train' des fichiers 
    #'nom_model_scene_*_norm.txt'

    for inst_path in sorted(path_to_size):  #parcourt les fichiers .txt associés aux modèles d'objets du training set dans obj_models
        instance = os.path.basename(inst_path).split('.')[0]  #nom du fichier du modèle en cours (ex : bottle3_scene5_norm)
        bbox_dims = np.loadtxt(inst_path)  #array numpy 1D à 3 éléments qui contient les dimensions physiques de la bbox du modèle d'objet en cours
        scale_factors[instance] = np.linalg.norm(bbox_dims)  #ajoute l'échelle de l'objet au dict des échelles pour le modèle de l'objet en cours

    # meta info for re-label mug category
    with open(os.path.join(data_dir, 'obj_models/mug_meta.pkl'), 'rb') as f:
        mug_meta = cPickle.load(f)

    valid_img_list = []  #liste des préfixe de path (type  'train/scene_*/****'') des images du test set qui sont valides (n'ont pas d'objets erronés)
    for img_path in tqdm(real_train):  #parcourt les préfixes de paths des images du train set (ex : 'train/scene_*/****')
        img_full_path = os.path.join(data_dir, 'Real', img_path)  #path de type 'data/Real/train/scene_*/****'
        all_exist = os.path.exists(img_full_path + '_color.png') and \
                    os.path.exists(img_full_path + '_coord.png') and \
                    os.path.exists(img_full_path + '_depth.png') and \
                    os.path.exists(img_full_path + '_mask.png') and \
                    os.path.exists(img_full_path + '_meta.txt')  #booléen qui indique si les éléments color.png, coord.png, depth.png, mask.png, meta.txt sont présents 
                        #pour chaque image du train set
        if not all_exist:  #si un des éléments est manquant pour l'image en cours
            continue  #ignore l'image

        depth = load_depth(img_full_path)  #array numpy 2D, carte de profondeur de l'image en cours
        masks, coords, class_ids, instance_ids, model_list, bboxes = process_data(img_full_path, depth)  #masks : array numpy 3D masques binaires de chaque objet dans 
        #l'image, class_ids : liste des ids des catégories des objets dans l'image, bboxes : array numpy 2D  bbox de chaque objet dans l'image (y1, x1, y2, x2)

        if instance_ids is None:  #s'il y a eu une erreur dans le rendering ou qu'il n'y a pas d'objet dans la scène
            continue  #passe à l'image suivante
        
        #calcule la pose
        num_insts = len(class_ids)  #nombre d'objets dans l'image
        scales = np.zeros(num_insts)  #array numpy 1D, contient pour chaque objet dans l'image son facteur d'échelle
        rotations = np.zeros((num_insts, 3, 3))  #array numpy 3D (n,3,3), contient pour chaque objet dans l'image sa matrice de rotation
        translations = np.zeros((num_insts, 3))  #array numpy 2D (n,3), contient pour chaque objet dans l'image sa translation
        
        for i in range(num_insts):    #parcourt les objets dans l'image enregistrées dans class_ids/masks/coords/instance_ids/model_list/bboxes 
            s = scale_factors[model_list[i]]  #échelle de l'objet
            mask = masks[:, :, i]  #array numpy 2D, masque binaire de l'objet
            idxs = np.where(mask)  #tuple d'arrays numpy 1D qui contiennent les indices de mask en lignes et en colonnes où les pixels appartiennent à l'objet
            coord = coords[:, :, i, :]  #map NOCS de l'ojet dans l'image
            coord_pts = s * (coord[idxs[0], idxs[1], :] - 0.5)
            coord_pts = coord_pts[:, :, None]   #Array of object points in the object coordinate space, Nx3 1-channel or 1xN/Nx1 3-channel, where N is the number of points
            img_pts = np.array([idxs[1], idxs[0]]).transpose()  #array numpy 2D (n,2), contient les indices des colonnes de mask où les pixels appartiennent à l'objet dans
            #la première colonne, indices des lignes de mask où les pixels appartiennent à l'objet dans la deuxième
            img_pts = img_pts[:, :, None].astype(float)  #ajoute une dimension de 1 à la fin de l'array img_pts
            distCoeffs = np.zeros((4, 1))    #array numpy 2D (4,1), contient les coefficients de distorsion (ici nuls, pas de distorsions)
            
            retval, rvec, tvec = cv2.solvePnP(coord_pts, img_pts, intrinsics, distCoeffs)  #retval : booléen? , rvec : array 2D (3,1) qui contient le vecteur de rotation
            #estimée de l'objet, tvec : array qui contient la translation estimée de l'objet dans l'image. Calculés par résolution des correspondances de points 3D-2D.
            assert retval  #vérifie ??

            R, _ = cv2.Rodrigues(rvec)  #array numpy 2D (3,3), vecteur de rotation converti en matrice de rotation 
            T = np.squeeze(tvec)  #

            # re-label for mug category           
            if class_ids[i] == 6:
                T0 = mug_meta[model_list[i]][0]
                s0 = mug_meta[model_list[i]][1]
                T = T - s * R @ T0
                s = s / s0

            scales[i] = s  #échelle de l'instance i dans l'image en cours
            rotations[i] = R  #matrice de rotation de l'instance i dans l'image en cours
            translations[i] = T  #vecteur de translation de l'instance i dans l'image en cours

        # write results
        gts = {}
        gts['class_ids'] = class_ids  #array numpy 1D d'ints qui contient la liste des ids des catégories des objets dans l'image
        gts['bboxes'] = bboxes  #array numpy 2D (n,4), contient la bbox de chaque objet dans l'image (y1, x1, y2, x2)
        gts['scales'] = scales.astype(np.float32)  # np.array, scale factor from NOCS model to depth observation
        gts['rotations'] = rotations.astype(np.float32)    #array numpy 3D (n,3,3), contient pour chaque objet dans l'image sa matrice de rotation
        gts['translations'] = translations.astype(np.float32)  #array numpy 2D (n,3), contient pour chaque objet dans l'image sa translation
        gts['instance_ids'] = instance_ids   #array numpy 1D d'ints qui contient la liste des numéros des objets dans l'image
        gts['model_list'] = model_list  #liste de strings qui contient les noms des modèles des objets dans l'image


        with open(img_full_path + '_label.pkl', 'wb') as f: #enregistre dans un path type data/Real/train/scene_*/**** le dict gts associé à l'image en cours avec le nom
        #'data/Real/train/scene_*/****'
            cPickle.dump(gts, f)  #enregistre sous le nom '****_label.pkl' au path  'data/Real/train/scene_*/' le dict gts associé à l'image en cours
        valid_img_list.append(img_path)  #ajoute le préfixe du path de l'image (type train/scene_*/****) à la liste des images valides


    # write valid img list to file
    with open(os.path.join(data_dir, 'Real/train_list.txt'), 'w') as f:   #ouvre le fichier au path 'data/Real/train_list.txt'
        for img_path in valid_img_list:
            f.write("%s\n" % img_path)  #pour chaque préfixe de path type train/scene_*/**** correspondant à une image valide, l'écrit dans le fichier ouvert



def annotate_test_data(data_dir):
    """ Generate gt labels for test data.
        Properly copy handle_visibility provided by NOCS gts.
        Enregistre dans Real/test pour chaque image de chaque scène un fichier '****_label.pkl' qui contient un dictionnaire de clefs/valeurs :
            - 'class_ids' : array numpy 1D d'ints qui contient la liste des ids des catégories des objets dans l'image
            - 'instance_ids' : array numpy 1D d'ints qui contient la liste des numéros des objets dans l'image
            - 'model_list' : liste de strings qui contient les noms des modèles des objets dans l'image
            - 'size' : array numpy 2D (n,3) où n nombre d'instances dans l'image, contient pour chaque objet dans l'image ses dimensions 3D (ou celles de son modèle NOCS?)
            - 'scales' : array numpy 1D avec pour chaque objet dans l'image son scale factor from NOCS model to depth observation
            - 'rotations' : array numpy 3D (n,3,3), contient pour chaque objet dans l'image sa matrice de rotation
            - 'translations' : array numpy 2D (n,3), contient pour chaque objet dans l'image sa translation
            - 'poses' : array numpy 3D (n,3,3), contient pour chaque objet dans l'image sa rotation, la translation + 1 ligne (cf gt_Rts)
            - 'bboxes' : array numpy 2D (n,4), contient la bbox de chaque objet dans l'image (y1, x1, y2, x2)
            - 'handle_visibility' : uniquement pour CAMERA
            
    """
    # Statistics:
    # test_set    missing file     bad rendering    no (occluded) fg    occlusion (< 64 pts)
    #   val        3792 imgs        132 imgs         1856 (23) imgs      50 insts
    #   test       0 img            0 img            0 img               2 insts

    # camera_val = open(os.path.join(data_dir, 'CAMERA', 'val_list_all.txt')).read().splitlines()  #fichier de lignes  type val/02445/0008
    real_test = open(os.path.join(data_dir, 'Real', 'test_list_all.txt')).read().splitlines()  #fichier de lignes type test/scene_6/0064
    # camera_intrinsics = np.array([[577.5, 0, 319.5], [0, 577.5, 239.5], [0, 0, 1]])  #paramètres intrinsèques de la caméra pour le dataset CAMERA
    real_intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])  #paramètres intrinsèques de la caméra pour le dataset real
    


    # compute model size
    model_file_path = ['obj_models/real_test.pkl']  #deux objets enregistrés avec Pickle dans le répertoire obj_models, créés par
    #la fonction 'save_nocs_model_to_file' du fichier shape_data.py. Contiennent un dict qui pour chaque modèle contenu dans un fichier .obj a un nuage de points générés
    #aléatoirement sur sa surface en valeur, stocké dans un array numpy. Uniquement pour les tests sets de CAMERA et Real

    models = {}  #dict qui contient pour chaque fichier .obj des tests sets de CAMERA et Real un nuage de points généré aléatoirement sur la surface de l'objet. En clefs
    #les noms des fichiers .obj, en valeurs des arrays numpy 2D (1024,3) contenant les coordonnées 3D (normalisées par l'échelle de l'objet) des 1024 points générés sur 
    #la surface de l'objet

    for path in model_file_path:
        with open(os.path.join(data_dir, path), 'rb') as f:
            models.update(cPickle.load(f))  #concatène le dict contenu dans le fichier .pkl ouvert au dict models
    model_sizes = {}  #dict qui contient pour chaque fichier .obj des tests sets de CAMERA et Real un array numpy 1D à 3 éléments qui contient la taille de l'objet (dim physiques)
    

    for key in models.keys():  #parcourt les fichiers .obj des tests sets de CAMERA et Real
        model_sizes[key] = 2 * np.amax(np.abs(models[key]), axis=0)  #calcule la taille (dimension physique) de l'objet



    # meta info for re-label mug category
    with open(os.path.join(data_dir, 'obj_models/mug_meta.pkl'), 'rb') as f:
        mug_meta = cPickle.load(f) 

    subset_meta = [('Real', real_test, real_intrinsics, 'test')]  #liste à 2 éléments, chacun est un tuple pour un subset de 
    #CAMERA et Real. Chaque tuple contient 4 éléments : le nom du dataset, la liste des images du subset, les paramètres intrinsèques de la caméra et le subset présent (test/val)

    for source, img_list, intrinsics, subset in subset_meta:

        valid_img_list = []  #liste des préfixe de path (type test/scene_*/**** ou val/*****/****) des images du test set qui sont valides (n'ont pas d'objets erronés)
        for img_path in tqdm(img_list):  #parcourt la liste des images du subset

            valid = True  #modifié, permet de supprimer une image qui contient une instance dont le hash a été supprimé dans shape_data (~180 images/27000)

            img_full_path = os.path.join(data_dir, source, img_path)  #path de type data/Real/test/scene_6/0064 ou data/CAMERA/val/02445/0008
            all_exist = os.path.exists(img_full_path + '_color.png') and \
                        os.path.exists(img_full_path + '_coord.png') and \
                        os.path.exists(img_full_path + '_depth.png') and \
                        os.path.exists(img_full_path + '_mask.png') and \
                        os.path.exists(img_full_path + '_meta.txt')  #booléen qui indique si les éléments color.png, coord.png, depth.png, mask.png, meta.txt sont présents 
                        #pour chaque image du subset

            if not all_exist:  #si un élément manque pour l'image en cours
                continue  #passe à l'image suivante

            depth = load_depth(img_full_path)  #array numpy 2D, carte de profondeur de l'image en cours
            masks, coords, class_ids, instance_ids, model_list, bboxes = process_data(img_full_path, depth)  #masks : array numpy 3D masques binaires de chaque objet dans 
            #l'image, class_ids : liste des ids des catégories des objets dans l'image, bboxes : array numpy 2D  bbox de chaque objet dans l'image (y1, x1, y2, x2)


            if instance_ids is None:  #s'il y a eu une erreur dans le rendering ou qu'il n'y a pas d'objet dans la scène
                continue  #passe à l'image suivante
            num_insts = len(instance_ids)  #nombre d'objets dans l'image

            # match each instance with NOCS ground truth to properly assign gt_handle_visibility
            #nocs_dir = os.path.join(os.path.dirname(data_dir), 'results/nocs_results')  #Erreur ici, le path ../results/nocs_results n'existe pas
            nocs_dir = os.path.join(data_dir, 'gts')  #path data/gts/

            if source == 'CAMERA':
                nocs_path = os.path.join(nocs_dir, 'val', 'results_val_{}_{}.pkl'.format(
                    img_path.split('/')[-2], img_path.split('/')[-1]))
            else:
                #nocs_path = os.path.join(nocs_dir, 'real_test', 'results_test_{}_{}.pkl'.format(img_path.split('/')[-2], img_path.split('/')[-1])) 
                nocs_path = os.path.join(nocs_dir, 'real_test', 'results_real_test_{}_{}.pkl'.format(img_path.split('/')[-2], img_path.split('/')[-1])) 
                #modifié, les fichiers results_test_scene_x_xxxx.pkl n'existent pas , le bon nom de fichier est results_real_test_scene_x_xxxx.pkl


            with open(nocs_path, 'rb') as f: 
                nocs = cPickle.load(f)  #dict avec comme clefs 'image_path', 'gt_RTs' et (uniquement pour CAMERA) 'gt_bboxes', 'gt_class_ids', 'handle_visibility',
                #'obj_list', 'gt_scales', 'image_id'


            if source == 'CAMERA' :  #modifié, ces clefs ne sont pas présentes pour le dataset Real
                gt_class_ids = nocs['gt_class_ids']  #liste des ids des classes (entre 1 et 6) des objets dans l'image
                gt_bboxes = nocs['gt_bboxes']
            
            gt_sRT = nocs['gt_RTs']  #array numpy (n,4,4) où n nombre d'instances dans l'image. Contient pour chaque instance (dim 1) la rotation, translation ground truth 
            #et une dernière ligne de binaires [0,0,0,1], cf https://www.brainvoyager.com/bv/doc/UsersGuide/CoordsAndTransforms/SpatialTransformationMatrices.html pour
            #explication 
            
            

            if source == 'CAMERA' :
                if 'handle_visibility' not in nocs.keys():  #rajouté car dans certains cas pour CAMERA, pas de gt_handle_visibility dans result
                    nocs['handle_visibility'] = np.ones_like(nocs['gt_class_ids'])  #rajouté en suivant evaluate, arrays de 1 de la même taille que gt_class_ids            
                gt_handle_visibility = nocs['handle_visibility']  #array numpy 1D qui contient autant d'éléments que d'instances dans l'image. Ses éléments
                #sont des floats : 0 ou 1 selon si on doit gérer la visibilité du mug (?) 


            map_to_nocs = []  #liste d'indices de gt_class_ids tels que gt_class_ids[map_to_nocs[i]] = class_ids[i] ie correspondances entre les instances de l'image
            #enregistrées dans gt_class_ids et les instances de l'image enregistrées dans class_ids/masks/coords/instance_ids/model_list/bboxes

            if source == 'CAMERA' :  #modifié, ajouté car pour Real il n'y a pas les clefs nécessaires dans nocs
                for i in range(num_insts): #parcourt les instances de l'image enregistrées dans class_ids/masks/coords/instance_ids/model_list/bboxes 
                    gt_match = -1  
                    for j in range(len(gt_class_ids)):  #parcourt les instances de l'image enregistrées dans gt_class_ids/gt_sRT
                        if gt_class_ids[j] != class_ids[i]:  #si la classe ground-truth dans gt_class_ids ne correspond pas à la classe du masque dans class_ids
                            continue  #passe au label ground truth suivant
                        if np.sum(np.abs(bboxes[i] - gt_bboxes[j])) > 5:  #si la différence entre les coordonnées des bounding boxes ground-truth et celle des masques gt pour tous les objets
                        #est supérieure à 5
                            continue  #passe au label ground truth suivant
                        # match found
                        gt_match = j  #si aucune des deux exceptions n'intervient, l'instance du masque ground truth correspond à celle dans l'image
                        break
                    # check match validity
                    assert gt_match > -1, print(img_path, instance_ids[i], 'no match for instance')  #si aucun match n'a été trouvé pour l'instance i
                    assert gt_match not in map_to_nocs, print(img_path, instance_ids[i], 'duplicate match')  #si l'instance matche 2 fois
                    map_to_nocs.append(gt_match)  #ajoute le numéro de l'instance dans l'image

         
                
            
            # copy from ground truth, re-label for mug category
            if source == 'CAMERA' :  #modifié, clef absente pour Real                
                handle_visibility = gt_handle_visibility[map_to_nocs]  #ne conserve que les instances effectivement matchées avec des masques pour handle_visibility
            
            sizes = np.zeros((num_insts, 3))  #array numpy 2D qui contient pour chaque instance présente dans l'image sa dimension physique en 3D
            poses = np.zeros((num_insts, 4, 4))  #array numpy 3D qui contient pour chaque instance dans l'image sa matrice qui contient sa rotation et translation 
            #(+ 1 ligne, cf 'gt_RTs' + haut)

            scales = np.zeros(num_insts)  #array numpy 1D, contient les échelles des instances dans l'image 
            rotations = np.zeros((num_insts, 3, 3))  #array numpy 3D qui contient la matrice de rotation de chaque instance dans l'image
            translations = np.zeros((num_insts, 3))  #array numpy 2D le vecteur de translation de chaque instance dans l'image
            
            for i in range(num_insts):  #parcourt les instances de l'image enregistrées dans class_ids/masks/coords/instance_ids/model_list/bboxes 

                if source == 'CAMERA' : #modifié, map_to_nocs pas créé pour Real
                    gt_idx = map_to_nocs[i]  #indice gt_sRT qui correspond au i-è objet enregistré dans class_ids/masks/coords/instance_ids/model_list/bboxes 


                if model_list[i] == 'd3b53f56b4a7b3b3c9f016d57db96408' :  #modifié, permet de supprimer les images où une instance est associée au hash supprimé dans shape_data
                    valid = False #permet d'indiquer qu'il ne faut pas enregistrer l'image
                    continue

                sizes[i] = model_sizes[model_list[i]]  #array numpy 1D à 3 éléments taille (physique) de l'objet dans l'image

                if source == 'CAMERA' :  #modifié
                    sRT = gt_sRT[gt_idx]  #matrice 4x4 qui contient la rotation, la translation + 1 ligne (cf gt_Rts + haut)
                else :  #pour Real, gt_idx n'est pas calculé, on suppose que les instances dans l'image sont ordonnées
                    sRT = gt_sRT[i]

                s = np.cbrt(np.linalg.det(sRT[:3, :3]))  #échelle de l'instance, dilatation de la matrice de rotation (normalement le déterminant est = à 1). 
                #On utilise racine cubique car determinant d'une matrice 3x3 calculé par addition de multiplication de 3 termes (donc dilatation^3)

                R = sRT[:3, :3] / s  #matrice de rotation de l'instance en enlevant la dilatiation
                T = sRT[:3, 3]  #vecteur de translation de l'instance
                
                # re-label mug category
                if class_ids[i] == 6:
                    T0 = mug_meta[model_list[i]][0]
                    s0 = mug_meta[model_list[i]][1]
                    T = T - s * R @ T0
                    s = s / s0

                # used for test during training
                scales[i] = s  #échelle de l'instance i dans l'image en cours
                rotations[i] = R  #matrice de rotation de l'instance i dans l'image en cours
                translations[i] = T  #vecteur de translation de l'instance i dans l'image en cours
                # used for evaluation
                sRT = np.identity(4, dtype=np.float32)
                sRT[:3, :3] = s * R  #ré applique l'échelle à la rotation
                sRT[:3, 3] = T
                poses[i] = sRT  #matrice 4x4 qui contient la rotation, la translation + 1 ligne (cf gt_Rts + haut)

            # write results
            gts = {}
            gts['class_ids'] = np.array(class_ids)  #array numpy 1D d'ints qui contient la liste des ids des catégories des objets dans l'image
            gts['instance_ids'] = instance_ids  #array numpy 1D d'ints qui contient la liste des numéros des objets dans l'image 
            gts['model_list'] = model_list    #liste de strings qui contient les noms des modèles des objets dans l'image
            gts['size'] = sizes   #array numpy 2D (n,3) où n nombre d'instances dans l'image, contient pour chaque objet dans l'image ses dimensions 3D (ou celles de son modèle NOCS?)
            gts['scales'] = scales.astype(np.float32)    # array numpy 1D avec pour chaque objet dans l'image son scale factor from NOCS model to depth observation
            gts['rotations'] = rotations.astype(np.float32)    #array numpy 3D (n,3,3), contient pour chaque objet dans l'image sa matrice de rotation
            gts['translations'] = translations.astype(np.float32)   #array numpy 2D (n,3), contient pour chaque objet dans l'image sa translation
            gts['poses'] = poses.astype(np.float32)    #array numpy 3D (n,3,3), contient pour chaque objet dans l'image sa rotation, la translation + 1 ligne 
            #(cf gt_Rts + haut)
            gts['bboxes'] = bboxes    #array numpy 2D (n,4), contient la bbox de chaque objet dans l'image (y1, x1, y2, x2)

            if source == 'CAMERA' :  #modifié, ces éléments non dispos pour Real                          
                gts['handle_visibility'] = handle_visibility    # handle visibility of mug
            
            with open(img_full_path + '_label.pkl', 'wb') as f:  #enregistre dans un path type data/Real/test/scene_6/0064 ou data/CAMERA/val/02445/0008 le dict gts associé 
            #à l'image en cours avec le nom
                cPickle.dump(gts, f)  #enregistre sous le nom '****_label.pkl' au path  'data/Real/test/scene_*/' ou 'data/CAMERA/val/*****/'' le dict gts associé à l'image
                #en cours
            
            if valid :  #modifié, gère le cas où l'image contient un objet dont le hash correspond au hash supprimé dans shape_data 
                valid_img_list.append(img_path)  #ajoute le préfixe du path de l'image (type test/scene_*/**** ou val/*****/****) à la liste des images valides



        
        # write valid img list to file
        with open(os.path.join(data_dir, source, subset+'_list.txt'), 'w') as f:  #ouvre le fichier au path 'data/Real/test_list.txt' ou 'data/CAMERA/val_list.txt'
            for img_path in valid_img_list:  
                f.write("%s\n" % img_path)  #pour chaque préfixe de path type test/scene_*/**** ou val/*****/**** correspondant à une image valide, l'écrit dans le fichier
            #ouvert




if __name__ == '__main__':
    data_dir = '../data'
    # create list for all data
    create_img_list(data_dir)
    # annotate dataset and re-write valid data to list
    #annotate_camera_train(data_dir)
    annotate_real_train(data_dir)
    annotate_test_data(data_dir)
