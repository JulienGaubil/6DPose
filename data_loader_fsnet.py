# @Time    : 25/09/2020 18:02
# @Author  : Wei Chen
# @Project : Pycharm
import torch
from torch.utils.data import Dataset, DataLoader
import _pickle as pickle
from uti_tool import *
import random


def getFiles(file_dir,suf):
    L=[]
    for root, dirs, files in os.walk(file_dir):
        #print('root: ',dirs)
        for file in files:
            if os.path.splitext(file)[1] == suf:
                L.append(os.path.join(root, file))
        L.sort(key=lambda x:int(x[-11:-4]))
    return L

def getDirs(file_dir):
    L=[]

    dirs = os.listdir(file_dir)

    return dirs


def load_depth(depth_path):
    """ Load depth image from img_path. """

    depth = cv2.imread(depth_path, -1)
    if len(depth.shape) == 3:
        # This is encoded depth image, let's convert
        # NOTE: RGB is actually BGR in opencv
        depth16 = depth[:, :, 1]*256 + depth[:, :, 2]
        depth16 = np.where(depth16==32001, 0, depth16)
        depth16 = depth16.astype(np.uint16)
    elif len(depth.shape) == 2 and depth.dtype == 'uint16':
        depth16 = depth
    else:
        assert False, '[ Error ]: Unsupported depth type.'
    return depth16


def chooselimt(pts0, lab, zmin, zmax):


    pts = pts0.copy()
    labs = lab.copy()

    pts1=pts[np.where(pts[:,2]<zmax)[0],:]
    lab1 = labs[np.where(pts[:,2]<zmax)[0], :]

    ptsn = pts1[np.where(pts1[:, 2] > zmin)[0], :]
    labs = lab1[np.where(pts1[:, 2] > zmin)[0],:]

    return ptsn,labs

def circle_iou(pts,lab, dia):
    # fx = K[0, 0]
    # ux = K[0, 2]
    # fy = K[1, 1]
    # uy = K[1, 2]
    a = pts[lab[:, 0] == 1, :]
    ptss = pts[lab[:, 0] == 1, :]
    idx = np.random.randint(0, a.shape[0])

    zmin = max(0,ptss[idx,2]-dia)
    zmax = ptss[idx,2]+dia

    return zmin, zmax



#classe qui est le dataset correspondant à une catégorie 'cate' précise
class CateDataset(Dataset):


    '''
    Input : 
        - root_dir : string, path du répertoire où se situent les répertoires des catégories
        - K : array numpy (3,3), paramètres intrinsèques de la caméra
        - cate : string, nom d'une des catégories du dataset
        - lim : int, utilisé dans la déformation de données
        - transform : ? utilisé dans la déformation de données 
        - corners : ? utilisé dans la déformation de données
        - temp : ?
    '''
    def __init__(self, root_dir, K, cate,lim=1,transform=None,corners=0, temp=None):  #cate : string qui donne une des catégories du dataset, K : paramètres intrinsèques de la caméra (?)

        cats = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']  #à modifier, spécifique à NOCS

        objs = os.listdir(root_dir)  #liste des entrées dans le directory root_dir
        self.objs_name = objs  #liste des entrées dans le directory root_dir
        self.objs = np.zeros((len(objs),1),dtype=np.uint)  #array numpy de zéros 2D (n,1) où n nombre d'entrées dans root_dir

        for i in range(len(objs)):  #parcourt la liste des entrées dans root_dir
            if cate in objs[i]:  #si la string cate de la catégorie est contenue dans la string objs[i] qui est le nom de la i-ème entrée du directory root_dir
                self.objs[i]=1  #met un 1 à la i-è ligne de l'array objs

        self.cate_id = np.where(np.array(cats)==cate)[0][0]+1  #indice de la catégorie cate dans la liste de catégories
        self.ids = np.where(self.objs==1)  #array numpy d'indices de self.objs où cate est une entrée dans root_dir
        self.root_dir = root_dir
        self.lim=lim  #?
        self.transform=transform  #?
        self.cate = cate  #catégorie du dataset
        self.K = K  #paramètres intrinsèques de la caméra?
        self.corners = corners  #?
        self.rad=temp  #?
        if cate=='laptop':  #modifié, précédemment labtop
            self.rad = 600
        if cate == 'bottle':
            self.rad = 400


        #pour NOCS, à modifier
        datapath = 'data/Real/train/scene_' #modifié, avant Real/train/scene
        model_path = 'data/obj_models/real_train/plys/'  #modifié, avant real_train/plys/


        self.data = datapath  #path dues fichiers des scènes d'entraînement
        self.c = random.randint(0, len(self.ids) - 1)  #Faux ? int choisi aléatoirement entre 0 et 1. self.ids tuple, mais devrait être len(self.ids[0])-1 ?
        self.model_path = model_path  #path des fichiers .ply des modèles des objets
    
    def __len__(self):


        return 1500 #à modifier?

    '''
    Input : 
        - index : clef appelée après un objet CateDataset (ex a['index']). Inutile
    Output : 
        - sample : dict de clefs :
            -> points : array numpy, ?
            -> label : array numpy, ?
            -> R : array numpy de taille (3,3) qui contient la rotation d'un des objets de catégorie 'cate' au sein d'une image choisie aléatoirement parmi les
            #scènes. Rotation valable pour toutes les données déformées.
            -> T : array numpy de taille (1,3) qui contient la translation d'un des objets de catégorie 'cate' au sein d'une image choisie aléatoirement parmi les
            #scènes. Rotation valable pour toutes les données déformées.
            -> cate_id : indice de la catégorie 'cate'
            -> scale : array numpy qui contient l'échelle (valable pour toutes les données déformées ou plusieurs scales contenues?)
            -> dep : string, contient le path vers la carte de profondeur de l'image choisie aléatoirement parmi les scènes.
    '''

    def __getitem__(self, index):  #methode qui définit ce que renvoie un objet 'a' de la classe appelé avec une clef 'index' (comme un dict, a[index])


        c = random.randint(0, len(self.ids[0])-1)  #int choisi aléatoirement entre 0 et le nombre de fois que cate apparaît dans root_dir -1
        #devrait être la définition de self.c? 

        obj_id = self.ids[0][c]  #indice d'une des lignes (choisie au hasard) de objs pour laquelle cate est une entrée de root_dir
        cate = self.objs_name[obj_id]  #nom de l'entrée dans root_dir d'indice obj_id (éventuellement != self.cate si cate sous-string du nom de l'entrée )

        pc = load_ply(self.model_path+'/%s.ply'%(cate))['pts']*1000.0  #charge les coordonnées des sommets/vertices qui composent les faces polygonales qui composent le
        #modèle 3D de la catégorie cate



        #à revoir, fichiers .txt manquants pour Linemod...
        root_dir = self.root_dir + '/%s/' % (cate)  #path self.root_dir/cate/
        pts_ps = getFiles_ab(root_dir+'points/','.txt',-12,-4)  #à modifier  à modifier -12 et -4 selon la place des chiffres dans les noms de fichiers, liste triée des
        #paths relatifs des fichiers à partir de 'self.root_dir/cate/points' dont l'extension est .txt. 
        idx = random.randint(0, len(pts_ps) - 1)  #int choisi aléatoirement entre 0 et le nombre de fichiers .txt
        pts_name = pts_ps[idx]  #nom d'un fichier txt choisi aléatoirement parmi ceux dans l'arborescence 'self.root_dir/cate/points'
        lab_name = getFiles_ab(root_dir+'points_labs/','.txt',-12,-4)[idx]  # à modifier -12 et -4 selon la place des chiffres dans les noms de fichiers
        #nom d'un des fichiers textes à partir du path 'self.root_dir/cate/points_labs/' dont l'extension est .txt, choisi aléatoirement



        scene_id = int(pts_name[-12:-4])//1000+1 #id de la scène dont est issue le fichier tirer aléatoirement, peut être modifié selon nos propres règles de naming
        img_id = int(pts_name[-12:-4])-(scene_id-1)*1000  #id de l'image dans la scène dont est issue le fichier tirer aléatoirement, peut être modifié selon nos propres règles
        #de naming

        depth_p  = self.data+'%d'%(scene_id)+'/%04d_depth.png'%(img_id)  #path de la map de profondeur du fichier choisi aléatoirement ( par ex'self.data7/0000_depth.png' 
        #avec self.data par égal à Real/train/scene_) 
        

        label_p = self.data+'%d'%(scene_id)+'/%04d_label.pkl'%(img_id)  #path d'un objet python enregistré qui contient la ground-truth du fichier choisi aléatoirement
        #( par ex'self.data7/0000_depth.png' avec self.data par égal à Real/train/scene_) 

        gts = pickle.load(open(label_p, 'rb'))  #ground-truth du fichier choisi aléatoirement chargée? objet dict de clefs 'model_list', 'bboxes', 'rotations', 'translations'
        idin = np.where(np.array(gts['model_list']) == cate)  #tuple de 2 arrays qui contiennent les indices où l'objet de catégorie cate est présent dans l'image choisie
        #aléatoirement


        if len(idin[0])==0: #répare les cas d'erreurs, si la catégorie cate n'est pas présente dans l'image choisie aléatoirement
            bbx = np.array([1,2,3,4]).reshape((1, 4))  #array numpy 2D de taille (1,4) qui contient une 
            R = np.eye(3)  #array numpy 2D de taille (3,3) qui contient une rotation identité
            T = np.array([0,0,0]).reshape(1,3)  #array numpy 2D de taille (1,3) qui contient une translation nulle
        else:  #si la catégorie cate est effectivement présente dans l'image choisie aléatoirement
            bbx = gts['bboxes'][idin[0]].reshape((1, 4)) ##bounding box sous la forme y1 x1 y2 x2 du premier objet de catégorie cate dans l'image choisie aléatoirement
            R = gts['rotations'][idin[0]].reshape(3,3)  #array numpy 2D de taille (3,3), rotation du premier objet de catégorie cate dans l'image choisie aléatoirement
            T = gts['translations'][idin[0]].reshape(1,3)*1000.0   #array numpy 2D de taille (1,4), translation du premier objet de catégorie cate dans l'image choisie
            #aléatoirement


        self.pc = pc  #fichier .ply de la catégorie cate chargé par la fonction load_ply (uti_tool.py)
        self.R = R  #rotation d'un objet de la catégorie cate au sein d'une image choisie aléatoirement parmi les scènes
        self.T = T  #translation d'un objet de la catégorie cate au sein d'une image choisie aléatoirement parmi les scènes
        depth = cv2.imread(depth_p,-1)  #array numpy 2D, charge la carte de profondeur de l'image choisie aléatoirement parmi les scènes
        # pts_name = bpp + 'pose%08d.txt' % (idx)

        label = np.loadtxt(lab_name)  #array numpy qui charge le texte contenu du fichier .txt choisi aléatoirement au path 'self.root_dir/cate/points_labs/'
        label_ = label.reshape((-1, 1))  #array label redimensionné en un vecteur colonne
        points_ = np.loadtxt(pts_name)  #array numpy qui charge le texte contenu dans le fichier .txt choisi aléatoirement au path 'self.root_dir/cate/points'



        points_, label_,sx,sy,sz = self.aug_pts_labs(depth,points_,label_,bbx)  #mécanisme de déformation des données pour l'augmentation?
        #sx, sy, sz composantes du vecteur échelle ?
        Scale = np.array([sx,sy,sz])  #vecteur échelle 


        if  points_.shape[0]!=label_.shape[0]:
            print(self.root_dir[idx])

        choice = np.random.choice(len(points_), 2000, replace=True)  #array numpy 1D de taille 2000 composé d'entiers tirés aléatoirement entre 0 et
        #len(points_)
        points = points_[choice, :]  #sélection aléatoire de points issus de la déformation de données
        label = label_[choice, :]  #sélection de labels aléatoire issus de la déformation de données

        sample = {'points': points, 'label': label, 'R':R, 'T':T,'cate_id':self.cate_id,'scale':Scale,'dep':depth_p}  #dictionnaire qui contient les 
        #données issues de la déformation de données appliquée à une image choisie aléatoirement parmi les scènes.

        return sample



    '''
    Réalise l'augmentation de données à partir des données d'un objet ground-truth.
    Input : 
        - depth : array numpy 2D qui contient la carte de profondeur de l'image
        - pts : array numpy qui contient ??
        - labs : array numpy 2D colonne de taille (n,1) qui contient les labels des objets présents dans l'image (?)
        - bbx : array numpy (?) qui contient les 4 indices de l'objet de la catégorie cate contenu dans l'image
    Output : 
        - sx, sy, sz : scalaires (?), composantes du vecteur échelle des déformations de l'objet de catégorie 'cate' dans l'image en entrée
        - 
    '''

    def aug_pts_labs(self, depth,pts,labs,bbx):

        ## 2D bounding box augmentation and fast relabeling
        bbx_gt = [bbx[0,1], bbx[0,3],bbx[0,0],bbx[0,2]]  #bounding box de l'objet de la catégorie cate présent dans l'image sous le format x1,x2, y1, y2
        bbx = shake_bbx(bbx_gt) ## x1,x2,y1,y2
        depth, bbx_iou = depth_out_iou(depth, bbx, bbx_gt)

        mesh = depth_2_mesh_bbx(depth, [bbx[2], bbx[3], bbx[0], bbx[1]], self.K)
        mesh = mesh[np.where(mesh[:, 2] > 0.0)]
        mesh = mesh[np.where(mesh[:, 2] < 5000.0)]

        if len(mesh) > 1000:
            choice = np.random.choice(len(mesh), len(mesh)//2, replace=True)
            mesh = mesh[choice, :]

        pts_a, labs_a = pts_iou(pts.copy(), labs.copy(), self.K, bbx_iou)

        assert pts_a.shape[0]==labs_a.shape[0]

        if len(pts_a[labs_a[:, 0] == 1, :])<50: ## too few points in intersection region
            pts_=pts_a.copy()
            labs_ = labs_a.copy()
        else:
            pts_ = pts.copy()
            labs_ = labs.copy()

        N = pts_.shape[0]
        M = mesh.shape[0]
        mesh = np.concatenate([mesh, pts_], axis=0)
        label = np.zeros((M + N, 1), dtype=np.uint)
        label[M:M + N, 0] = labs_[:, 0]
        points = mesh

        if self.lim == 1:
            zmin, zmax = circle_iou(points.copy(), label.copy(), self.rad)
            points, label = chooselimt(points, label,zmin, zmax)



        ### 3D deformation
        Rt = get_rotation(180,0,0)
        self.pc = np.dot(Rt, self.pc.T).T ## the object 3D model is up-side-down along the X axis in our case, you may not need this code to reverse


        s  = 0.8
        e = 1.2
        pointsn, ex,ey, ez,s = defor_3D(points,label, self.R, self.T, self.pc, scalex=(s, e),scalez=(s, e),
                                        scaley=(s, e), scale=(s, e), cate=self.cate)
        sx,sy,sz = var_2_norm(self.pc, ex, ey, ez, c=self.cate)
        return pointsn, label.astype(np.uint8), sx,sy,sz




'''
Crée un objet CateDataset et le wrap dans un dataloader
Input : 
    - data_path : string, path du répertoire où se situent les répertoires des catégories
    - bat : int, taille des batchs
    - K : array numpy de taille (3,3), paramètres intrisèques de la caméra
    - cate : string, nom d'une des catégories du dataset
    - lim : int, utilisé dans la déformation de données
    - rad : int, ?
    - shuf : bool, indique si on doit mélanger les données
    - drop : bool, ?
    - corners : int, ?
    - nw : int, nombre de workers utilisés par le dataloader
Output : 
    - dataloader : objet DataLoader qui wrap l'objet CateDataset
'''
def load_pts_train_cate(data_path ,bat,K,cate,lim=1,rad=400,shuf=True,drop=False,corners=0,nw=0):

    data=CateDataset(data_path, K, cate,lim=lim,transform=None,corners=corners, temp=rad)

    dataloader = DataLoader(data, batch_size=bat, shuffle=shuf, drop_last=drop,num_workers=nw)

    return dataloader










