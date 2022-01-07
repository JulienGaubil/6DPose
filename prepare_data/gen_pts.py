# @Time    : 12/05/2021
# @Author  : Wei Chen
# @Project : Pycharm

import cv2
import numpy as np
import os
import os.path
import sys
import _pickle as pickle
from tqdm import tqdm

sys.path.append('..')
from uti_tool import getFiles_cate, depth_2_mesh_all, depth_2_mesh_bbx
from renderer import create_renderer

def render_pre(model_path):
    renderer = create_renderer(640, 480, renderer_type='python', shading = 'flat')  #modifié, le type de shading flat n'utilise pas de normales pour les vertices

    #modifié, utilisation de la version modifiée de la fonction, getFiles
    # models = getFiles_ab_cate(model_path, 'ply') #modifié, ancien suffixe utilisé : '.ply', qui ne peut rien matcher. Liste des paths relatifs à model_path des fichiers .ply
    #trouvés dans l'arborescence

    model_paths, model_names = getFiles(model_path, 'ply')  #model_paths : liste des paths des fichiers .obj trouvés à partir du path model_path
    #model_names : liste des noms des fichiers .obj trouvés à partir du path obj_model_dir

    objs=[]  #liste des paths des fichiers .plys (sans les extensions '.plys')
    
    #modifié, parsing des noms incorrect
    # for model in models:  #parcourt les fichier .ply
    #     obj = model.split('.')[1]  # nom /obj_model/real_train/plys/obj_***, path du fichier .ply sans l'extension
    #     objs.append(obj)
    #     renderer.add_object(obj, model)


    for i in range (len(model_paths)):  #parcourt les fichiers .ply trouvés qui contiennent les modèles 3D d'objets
        obj_id = model_names[i].split('.')[0]  #string qui contient le nom du fichier .ply sans l'extension, sert d'id au modèle
        obj_path = model_paths[i]  #path du fichier .ply
        objs.append(obj_id)  #inutile?
        renderer.add_object(obj_id, obj_path)  #ajoute l'objet au FrameBuffer du renderer
    return renderer

def getFiles(file_dir,suf):  #fonction getFiles_ab_cate légèrement modifiée pour renvoyer les noms du fichiers
    '''
    Inputs :
        - file_dir : string, path à partir duquel chercher les fichiers 
        - suf : string, nom d'extension des fichiers à chercher (ex 'ply')
    Outputs : 
        - paths : liste des paths relatifs à file_dir des fichiers d'extension suf
        - names : liste des noms des fichiers d'extension suf trouvés (nom de fichier + extension)
    '''

    paths = []
    names = []
    for root, dirs, files in os.walk(file_dir):  #parcourt l'arborescence à partir de file_dir
        for file in files:  #parcourt la liste des fichiers à un endroit donné de l'arborescence
            if file.split('.')[1] == suf:  #si le fichier est un fichier dont l'extension correspond à suf
                paths.append(os.path.join(root, file))  #ajoute le path relatif du fichier à la liste
                names.append(file)  #ajoute le nom du fichier à la liste
    return paths, names

def getFiles_ab_cate(file_dir,suf):  
    '''
    Inputs :
        - file_dir : string, path à partir duquel chercher les fichiers 
        - suf : string, nom d'extension des fichiers à chercher (ex 'ply')
    Outputs : 
        - L : liste des paths relatifs à file_dir des fichiers d'extension suf
    '''

    L=[]
    for root, dirs, files in os.walk(file_dir):  #parcourt l'arborescence à partir de file_dir
        for file in files:  #parcourt la liste des fichiers à un endroit donné de l'arborescence
            if file.split('.')[1] == suf:  #si le fichier est un fichier dont l'extension correspond à suf
                L.append(os.path.join(root, file))  #ajoute le path relatif du fichier à la liste
    return L


def get_dis_all(pc,dep,dd=15):
    '''
    Inputs :
        - pc : array numpy 2D (n1*m1, 3), une bbox qui contient tous les pixels de profondeur valide (ie >0) est définie. array numpy 2D (n*m, 3) où n, m dimensions de la
                bbox définie par les pixels extrêmaux valides (profondeur>0) de la carte de profondeur. Contient pour chaque pixel de la bbox les coordonnées 3D du point correspondant
                dans le  référentiel de la caméra. (x,y,z)

        - dep : array numpy 2D (n2*m2, 3), où n2, m2 dimensions de la bbox définie par les pixels extrêmaux valides (profondeur>0) de la carte de profondeur autour d'un des 
                objets de l'image. Contient pour chaque pixel de la bbox les coordonnées 3D du point correspondant dans le référentiel de la caméra (x,y,z). 

        - dd : int, #distance maximale entre les points de pc et ceux de dep pour qu'on considère qu'ils sont identiques

    Output : 
        - ids : array numpy 1D, indices des lignes de dep qui contiennent des points qui sont dans pc (ou d'une distance inférieure à dd en norme 2)
    '''

    N=pc.shape[0]  #nombre de points de profondeur valide conservés dans la carte de profondeur de l'image totale
    M=dep.shape[0]  #nombre de points de profondeur valide conservés dans la carte de profondeur autour de l'objet
    
    depp=np.tile(dep,(1,N))  #array numpy 2D (M, N*dep.shape[1]), répète l'array dep N fois sur la dimension 2
    depmm=depp.reshape((M,N,3))  #array numpy 3D (M,N,3), contient dep répété N fois sur la dimension 2.  Le i-è array (N,3) (contenu à depmm[i, :, :]) contient N fois la
    #ligne i de dep.

    delta = depmm - pc    #array numpy 2D (M,N,3),diss.shape soustrait pc à depmm sur tous M les arrays 2D (N,3) de la première dim de depmm. Pour le i-è array (N,3),
    #soustrait pc à la i-è ligne de dep répétée N fois : compare la i-è ligne de dep avec toutes les lignes de pc

    diss=np.linalg.norm(delta,2, 2)  #array numpy 2D (M,N), contient pour chacun des M points de N sa distance en norme 2 avec chacun des N points contenus dans pc

    aa=np.min(diss,1)  #array numpy 1D à M éléments, conserve pour chaque point de dep sa distance avec le point de pc qui lui est le plus proche
    bb=aa.reshape((M,1))  #reshape aa en array 2D de shape (M,1)

    ids,cc=np.where(bb[:]<dd)  #ids array 1D qui contient les indices des lignes/colonnes de dep qui contiennent des points qui sont dans pc (suffisamment proches avec
    #une distance inférieure à dd)

    return ids


def get_one(depth, bbx, vispt, K, idx, objid, bp):
    '''
    Inputs : 
        - depth : array numpy 2D (h,w) où h,w dimensions d'une image. Carte de profondeur de image d'id idx
        - bbx : liste à 4 éléments, coordonnées de la bbox de l'objet dans l'image [y1, x1, y2, x2]
        - vispt : array numpy 2D (h,w) où h,w dimensions de l'image. Contient la carte de profondeur de l'image d'id idx avec l'objet d'id objid rendered/représenté dessus.
        - K : array numpy 2D (3,3), matrice des paramètres intrinsèques de la caméra
        - idx : int, id de l'image dans le dataset (ex : 0234 dans la scène 3 ==> 2234)
        - objid : string, nom du modèle de l'objet dans l'image. Doit matcher les noms des fichiers .ply qui définissent les modèles des objets ?
        - bp : string, path de base sous lequel sont enregistrés les points générés et les indices des points qui appartiennent à l'objet objid
                a priori  '../data/Real/train/pts'

    Output :
        - enregistre deux fichiers .txt qui contiennent des arrays numpy :
            -> au path bp + objid + '/points/pose********.txt' : array numpy 2D de shape (numbs, 3) qui contient les numbs points de la bouding box autour de l'objet 
                aléatoirement conservés (un point peut être répété). Les nombres ******** sont déterminés par l'id de l'image dans le dataset, idx.

            -> au path bp + objid + '/points_lab/lab********.txt' : array numpy 2D de shape (numbs, 1) qui contient des 0 et 1 pour les numbs points de la bbox autour de 
                l'objet aléatoirement conservés (un point peut être répété). 1 si le point appartient à l'objet d'id idx. Les nombres sont déterminés par l'id de l'image
                dans le dataset, idx.
    '''

    numbs = 6000  #nombre de points à générer au total

    numbs2 = 1000  #cf suite


    save_path = bp + '/%s/points' % (objid)  #path du dir où on enregistre les coordonnées 3D dans le référentiel de la caméra de numbs points aléatoirement sélectionnés
    #aléatoirement (éventuellement répétés) dans la bbox autour de l'objet d'objid, '../data/Real/train/pts/**...*/points'

    save_pathlab = bp + '/%s/points_labs' % (objid)  #path du dir où enregistre un array qui indique les indices des points enregistrés dans le dir save_path qui 
    #appartiennent effectivement à l'objet d'id objid ('../data/Real/train/pts/**...*/points_lab')

    if not os.path.exists(save_path):  #si le répertoire de path save_path n'existe pas
        os.makedirs(save_path)  #crée le répertoire de path save_path

    if not os.path.exists(save_pathlab):  #si le répertoire de path save_path n'existe pas
        os.makedirs(save_pathlab)  #crée le répertoire de path save_pathlab

    VIS = depth_2_mesh_all(vispt, K)  #une bbox qui contient tous les pixels de profondeur valide (ie >0) est définie. Array numpy 2D (N, 3) où N nombre de pixels dans la
    #bbox définie par les pixels extrêmaux valides (profondeur>0) de la carte de profondeur. Contient pour chaque pixel de la bbox les coordonnées 3D du point correspondant
    #dans le  référentiel de la caméra (x,y,z).

    VIS = VIS[np.where(VIS[:, 2] > 0.0)] * 1000.0  #ne conserve que les points où la profondeur est strictement positive, multiplie les coordonnées 3D par 1000

    if VIS.shape[0] > numbs2:  #s'il y a plus de numbs2 points de la carte de profondeur de l'image entière pour lesquels la profondeur est valide (>0)
        choice2 = np.random.choice(VIS.shape[0], numbs2, replace=False)  #array 1D de numbs2 ints choisis aléatoirement entre 0 et le nombre de points de profondeur valide
        VIS = VIS[choice2, :]  #échantillonne VIS aléatoirement en ne conservant que numbs2 points


    filename = save_path + ("/pose%08d.txt" % (idx))  #path du fichier où on enregistre enregistre les coordonnées 3D dans le référentiel de la caméra de numbs points 
    #sélectionnés aléatoirement (éventuellement répétés) dans la bbox autour de l'objet d'objid ('../data/Real/train/pts/**...*/points/pose********.txt')

    w_namei = save_pathlab + ("/lab%08d.txt" % (idx))  #path du fichier où on enregistre un array qui indique les indices des points enregistrés dans le dir save_path qui 
    #appartiennent effectivement à l'objet d'id objid ('../data/Real/train/pts/**...*/points_lab/lab********.txt')

    dep3d_ = depth_2_mesh_bbx(depth, bbx, K, enl=0)  #array numpy 2D (n*m, 3) où n, m dimensions de la bbox définie par les pixels extrêmaux valides (profondeur>0) de la 
    #carte de profondeur autour de l'objet d'id objid. Contient pour chaque pixel de la bbox les coordonnées 3D du point correspondant dans le référentiel de la caméra (x,y,z)

    if dep3d_.shape[0] > numbs:  #s'il y a plus de numbs points de la carte de profondeur autour de l'objet pour lesquels la profondeur est valide (>0)
        choice = np.random.choice(dep3d_.shape[0], numbs, replace=False)  #array 1D de numbs ints choisis aléatoirement entre 0 et le nombre de points de profondeur valide
        #sur la carte de profondeur autour de l'objet d'id objid. Les points sont présents une unique fois.

        dep3d = dep3d_[choice, :]  #échantillonne dep3d aléatoirement pour ne conserver que numbs points

    else:  #s'il y a moins de numbs points de la carte de profondeur autour de l'objet pour lesquels la profondeur est valide (>0)
        choice = np.random.choice(dep3d_.shape[0], numbs, replace=True)  #array 1D de numbs ints choisis aléatoirement entre 0 et le nombre de points de profondeur valide
        #sur la carte de profondeur autour de l'objet d'id objid. Certains points peuvent être répétés

        dep3d = dep3d_[choice, :]  #dep3d avec les points mélangés aléatoirement et certains répétés pour obtenir numb points

    dep3d = dep3d[np.where(dep3d[:, 2] != 0.0)]  #ne conserve que les points où la profondeur est strictement positive
    threshold = 12  #distance maximale entre les points de dep3d et ceux de VIS pour qu'on considère qu'ils sont identiques

    ids = get_dis_all(VIS, dep3d[:, 0:3], dd=threshold)  #array numpy 1D, indices des lignes de dep3d qui contiennent des points qui sont dans VIS
    #(ou d'une distance inférieure à threshold en norme 2). Ce sont les points qui appartiennent à l'objet d'id objid

    if len(ids) <= 10:  #si l'objet d'id objid est composé de moins de 10 points de profondeur valide
        if os.path.exists(filename):  #si le fichier ?? à enregistrer existe déjà
            os.remove(filename)  #supprime le fichier
        if os.path.exists(w_namei):  #si le fichier ?? à enregistrer existe déjà
            os.remove(w_namei)  #supprime le fichier

    if len(ids) > 10:  #si l'objet d'id objid est composé de plus de 10 points de profondeur valide
        np.savetxt(filename, dep3d, fmt='%f', delimiter=' ')  #enregistre l'array numpy 2D de shape (numbs, 3) qui contient les coordonnées 3D dans le référentiel de la
        #caméra des numbs points de la bounding box autour de l'objet aléatoirement conservés (un point peut être répété). Enregistre dans un fichier .txt de path filename
        #(typiquement '../data/Real/train/pts/**...*/points/pose********.txt', où les nombres sont déterminés par l'id de l'image dans le dataset, idx)

        lab = np.zeros((dep3d.shape[0], 1), dtype=np.uint)  #array numpy 2D de shape (numbs, 1) de zéros, contiendra des 1 aux indices des lignes dep3d qui correspondent
        #aux coordonnées d'un point qui appartient à l'objet. 

        lab[ids, :] = 1  #met des 1 aux lignes qui correspondent aux lignes de dep3d (points de profondeur valide (>0) sélectionnés aléatoirement dans la bbox autour de
        #l'objet d'id objid) qui contiennent des points qui appartiennent à l'objet d'id objid

        np.savetxt(w_namei, lab, fmt='%d')  #enregistre l'array numpy 2D de shape (numbs, 1) qui contient des 0 et 1 pour les numbs points de la bbox autour de 
        #l'objet aléatoirement conservés (un point peut être répété). 1 si le point appartient à l'objet d'id idx. Enregistre dans un fichier .txt de path w_namei
        #(typiquement '../data/Real/train/pts/**...*/points/pose********.txt', où les nombres sont déterminés par l'id de l'image dans le dataset, idx)



def get_point_wise_lab(basepath, fold, renderer, sp):


    '''
    Inputs : 
        - basepath : string, path 'data/Real/train/scene_'
        - fold : int, numéro de la scène dans laquelle aller chercher les ?? (ex : scene_5 ==> 5)
        - renderer : objet RenderPython renvoyé par render_pre?
        - sp : string, saving path?   a priori  '../data/Real/train/pts'

    Outputs : 
        - pour chaque image contenue dans la scène d'indice fold, pour chaque objet présent dans l'image, enregistre deux fichiers .txt qui contiennent des arrays numpy : 
        
            -> au path sp + obj + '/points/pose********.txt' : où obj id du modèle 3D de l'objet. Array numpy 2D de shape (N, 3) qui contient les coordonnées 3D dans le 
                référentiel de la caméra de N points de la bounding box autour de l'objet aléatoirement sélectionnés (un point peut être répété). Les nombres ******** sont
                déterminés par l'id de l'image dans le dataset

            -> au path sp + obj + '/points_lab/lab********.txt' : array numpy 2D de shape (N, 1) qui contient des 0 et 1 pour N points de la bbox autour de 
                l'objet aléatoirement sélectionnés (un point peut être répété). 1 si le point appartient à l'objet d'id idx. Les nombres sont déterminés par l'id de l'image
                dans le dataset.
    '''
    base_path = basepath + '%d/' % (fold)  #path basepath/fold


    depths = getFiles_cate(base_path, '_depth', 4, -4)  #liste des fichiers qui contiennent le suffixe '_depth' comme nom de fichiers. Liste triée par numéros de fichiers
    labels = getFiles_cate(base_path, '_label', 4, -4)  #modifié, pas de fichiers '_label2', remplacé par '_label'. Liste des fichiers qui contiennent le suffixe '_label2' 
    #comme nom de fichiers (fichiers .pkl). Liste triée par numéros de fichiers.


    L_dep = depths  #liste des fichiers qui contiennent le suffixe '_depth' comme nom de fichiers. Liste triée par numéros de fichiers
    Ki = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])  #paramètres intrinsèques de la caméra

    Lidx = 1000  #utilisé pour créer l'id de l'image
    if fold == 1:  #si la scène correspond à la scène 1
        s = 0  #on n'exclut aucune image de la scène
    else:  #si la scène n'est pas la scène 1
        s = 0  #on n'exclut aucune image de la scène

    for i in tqdm(range(s, len(L_dep))):  #parcourt les paths de toutes les cartes de profondeur trouvées dans la scène fold

        lab = pickle.load(open(labels[i], 'rb'))  #dictionnaire de clefs/ valeurs : 'model_list' : liste de strings, noms des modèles des objets dans l'image ; 
        #'rotations' : array numpy 3D (n,3,3), matrices de rotation des objets dans l'image ; 'translations' : array numpy 2D (n,3), translation des objets dans l'image ;
        #'bboxes' : array numpy 2D (n,4), bbox de chaque objet dans l'image (y1, x1, y2, x2) ; 'class_ids' : array numpy 1D d'ints, liste des ids des catégories des objets
        #dans l'image

        depth = cv2.imread(L_dep[i], -1)  #array numpy 2D, charge la carte de profondeur de l'image en cours
        img_id = int(L_dep[i][-14:-10])  #int, numéro de l'image dans la scène (ex : '0000_depth.png' ==> 0)
        
        for ii in range(len(lab['class_ids'])):  #parcourt les objets dans l'image correspondant à la carte de profondeur en cours

            obj = lab['model_list'][ii]  #string, nom du modèle de l'objet dans l'image, doit matcher les noms des fichiers .ply qui définissent les 
            #modèles des objets ?
            seg = lab['bboxes'][ii].reshape((1, 4))  #array numpy 2D (1,4), bbox de l'objet dans l'image (y1, x1, y2, x2)
            idx = (fold - 1) * Lidx + img_id  #id de l'image dans le dataset (ex : 0234 dans la scène 3 ==> 2234)
            R = lab['rotations'][ii]  # .reshape((3, 3))  #array numpy 2D (3,3), matrice de rotation de l'objet dans l'image
            T = lab['translations'][ii].reshape((3, 1))  #array numpy 2D (3,1), vecteur de translation de l'objet dans l'image

            if T[2] < 0:  #si la translation selon z est négative (ie en profondeur)
                T[2] = -T[2]  #erreur, remet la profondeur en positif

            vis_part = renderer.render_object(obj, R, T, Ki[0, 0], Ki[1, 1], Ki[0, 2], Ki[1, 2])['depth']  #array numpy 2D (h,w) où h,w dimensions de l'image. Contient la 
            #carte de profondeur de l'image avec l'objet d'id obj_id rendered/représenté dessus.

            bbx = [seg[0, 0], seg[0, 2], seg[0, 1], seg[0, 3]]  #liste à 4 éléments, coordonnées de la bbox de l'objet dans l'image [y1, x1, y2, x2]

            if vis_part.max() > 0:  #vérifie que les valeurs de la carte de profondeur avec l'objet rendered dessus sont bien positives (ie pas d'erreur)
                get_one(depth, bbx, vis_part, Ki, idx, obj, sp)  #enregistre deux fichiers .txt contenant des arrays numpy dans des sous-répertoires du path sp : 
            '''
            -> au path sp + obj + '/points/pose********.txt' : array numpy 2D de shape (N, 3) qui contient les coordonnées 3D dans le référentiel de la caméra de N points
                bounding box autour de l'objet aléatoirement conservés (un point peut être répété). Les nombres ******** sont déterminés par l'id de l'image dans le dataset, 
                idx.

            -> au path sp + obj + '/points_lab/lab********.txt' : array numpy 2D de shape (N, 1) qui contient des 0 et 1 pour les N points de la bbox autour de 
                l'objet aléatoirement conservés (un point peut être répété). 1 si le point appartient à l'objet d'id idx. Les nombres sont déterminés par l'id de l'image
                dans le dataset, idx.
            '''




if __name__ == '__main__':
    path_rend = '../data/obj_models/real_train/plys/'
    base_path_save_points = '../data/Real/train/pts'
    base_path_scenes = '../data/Real/train/scene_'  #préfixe du path où aller chercher les images/cartes de profondeur dans chaque scène


    scene_numbers = [1, 2, 3, 4, 5, 6, 7]  #numéro des scènes
    renderer = render_pre(path_rend)

    for num in tqdm(scene_numbers) : 
        get_point_wise_lab( base_path_scenes, num, renderer, base_path_save_points) #basepath, fold, renderer, sp







