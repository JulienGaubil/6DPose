import os
import os.path
import numpy as np
import pickle
from tqdm import tqdm






def getFiles(file_dir,suf):  
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




def load_obj(path_to_file):
    """ Charge un fichier .obj

    Input :
      - path_to_file: path du fichier .obj

    Output :
      - model : dict de clefs : 
        -> vertices: array numpy 2D de shape (n,3) où n nombre de vertices dans le fichier .obj. Contient les coordonnées 3D des vertices de l'objet.
        -> faces: array numpy 2D (m,3) où m nombre de faces dans le fichier .obj. Contient les indices des vertices qui composent chaque faces triangulaires.
        -> normals : optionnel, array numpy 2D (n,3) où n nombre de vertices dans le fichier .obj. Contient les coordonnées 3D des normales aux vertices.
    """
    vertices = [] 
    faces = []
    normals = []
    with open(path_to_file, 'r') as f:  #ouvre le fichier .obj
        for line in f:  #parcourt les lignes du fichier .obj
            
            if line[:2] == 'v ':  #cas où la ligne commence par la le pattern 'v ', ie ligne qui représente un vertex
                vertex = line[2:].strip().split(' ')  #liste contenant 3 strings qui sont les floats contenus dans la ligne en cours, coordonnées 3D du vertex
                vertex = [float(xyz) for xyz in vertex]  #convertit les éléments de vertex en floats
                vertices.append(vertex)  #ajoute le vertex représenté par la ligne en cours à la liste des vertices

            elif line[0] == 'f':  #cas où la ligne commence par la le pattern 'f', ie ligne qui représente une face
                face = line[1:].replace('//', '/').strip().split(' ')  #liste contenant 3 strings qui sont les indices (de type n/n où n int) contenus dans la ligne en
                #cours, indices des vertices qui composent la face (?) 
                face = [int(idx.split('/')[0])-1 for idx in face]  #convertit les éléments de face en ints (ne conserve que le premier int contenu dans la string des indices)
                faces.append(face)  #rajoute la face représentée par la ligne en cours à la liste des faces

            elif line[:3] == 'vn ':  #cas où la ligne commence par la le pattern 'vn ', ie ligne qui représente une normale
            	normal = line[3:].strip().split(' ')  #liste contenant 3 strings qui sont les floats contenus dans la ligne en cours, coordonnées 3D de la normale
            	normal = [float(xyz) for xyz in normal]  #convertit les éléments de normal en floats
            	normals.append(normal)   #ajoute la normale représentée par la ligne en cours à la liste des normales

            else:  #si la ligne ne correspond ni à un vertex ni à une face, ni à une normale, ignore la ligne
                continue
    
    vertices = np.asarray(vertices)  #convertit vertices en array
    faces = np.asarray(faces)  #convertit faces en array
    normals = np.asarray(normals)  #convertit normals en array

    if vertices.shape == (0,) and faces.shape == (0,) and normals.shape==(0,) :  #affiche le fichier .obj et son path dans le cas où le fichier ne contient aucun vertex ni aucune face
        print(path_to_file)
        print(open(path_to_file, 'r').readlines())

    model = {}  #crée le dict dans lequel sont stockés les arrays
    model['pts'] = vertices  #stocke les vertices
    model['faces'] = faces  #stocke les faces
    if len(normals) !=0 :  #vérifie que les normales sont bien présentes dans le fichier .obj
    	model['normals'] = normals  #stocke les normales

    return model













def save_ply(path, model, extra_header_comments=None):
  """Sauvegarde un modèle 3D en un fichier .ply.

  Inputs : 
    - path : string, path du fichier .ply qui sera enregistré (avec le nom de fichier et l'extension)
    - extra_header_comments : string, commentaires additionnels pour le header
    - model : dict, contient les infos du modèle 3D. Clefs (seule 'pts' est olbigatoire, les autres optionnelles) :

      -> 'pts' : array numpy nx3 où n nombre de vertices utilisés dans le fichier .ply pour décrire l'objet. Contient les coordonnées des vertices selon les axes x,y,z.
      -> 'normals' : array numpy nx3 où n nombre de vertices utilisés dans le fichier .ply pour décrire l'objet. Contient la normale aux faces associée à chaque vertex.
      -> 'colors' : array numpy nx3 où n nombre de vertices utilisés dans le fichier .ply pour décrire l'objet. Contient les triplets RGB associés à chaque vertex.          
      -> 'faces' : array numpy mx3 où m nombre de faces polygonales (triangulaires) utilisés dans le fichier .ply pour décrire l'objet. Contient les indices dans le
                  fichier .ply des 3 sommets/vertices qui composent chaque face polygonale          
      -> 'texture_uv_face' : array numpy mx6 où m nombre de faces triangulaires décrire l'objet. Propriétés de textures associées à chaque face.
      -> 'texture_file' : string, path relatif au fichier .ply de l'éventuel fichier texture associé au modèle.
      -> 'texture_uv' : array numpy nx2 où n nombre de vertices utilisés pour décrire l'objet. Coordonnées UV de texture associées à chaque vertex. Colonne 0 : u, 
                        Colonne 1 : v 
  """
  pts = model['pts']
  pts_colors = model['colors'] if 'colors' in model.keys() else None
  pts_normals = model['normals'] if 'normals' in model.keys() else None
  faces = model['faces'] if 'faces' in model.keys() else None
  texture_uv = model['texture_uv'] if 'texture_uv' in model.keys() else None
  texture_uv_face = model['texture_uv_face'] if 'texture_uv_face' in model.keys() else None
  texture_file = model['texture_file'] if 'texture_file' in model.keys() else None

  save_ply2(path, pts, pts_colors, pts_normals, faces, texture_uv,
            texture_uv_face,
            texture_file, extra_header_comments)


def save_ply2(path, pts, pts_colors=None, pts_normals=None, faces=None,
              texture_uv=None, texture_uv_face=None, texture_file=None,
              extra_header_comments=None):
  """Saves a 3D mesh model to a PLY file.

  :param path: Path to the resulting PLY file.
  :param pts: nx3 ndarray with vertices.
  :param pts_colors: nx3 ndarray with vertex colors (optional).
  :param pts_normals: nx3 ndarray with vertex normals (optional).
  :param faces: mx3 ndarray with mesh faces (optional).
  :param texture_uv: nx2 ndarray with per-vertex UV texture coordinates
    (optional).
  :param texture_uv_face: mx6 ndarray with per-face UV texture coordinates
    (optional).
  :param texture_file: Path to a texture image -- relative to the resulting
    PLY file (optional).
  :param extra_header_comments: Extra header comment (optional).
  """
  if pts_colors is not None:
    pts_colors = np.array(pts_colors)
    assert (len(pts) == len(pts_colors))

  valid_pts_count = 0
  for pt_id, pt in enumerate(pts):
    if not np.isnan(np.sum(pt)):
      valid_pts_count += 1

  f = open(path, 'w')
  f.write(
    'ply\n'
    'format ascii 1.0\n'
    # 'format binary_little_endian 1.0\n'
  )

  if texture_file is not None:
    f.write('comment TextureFile {}\n'.format(texture_file))

  if extra_header_comments is not None:
    for comment in extra_header_comments:
      f.write('comment {}\n'.format(comment))

  f.write(
    'element vertex ' + str(valid_pts_count) + '\n'
                                               'property float x\n'
                                               'property float y\n'
                                               'property float z\n'
  )
  if pts_normals is not None:
    f.write(
      'property float nx\n'
      'property float ny\n'
      'property float nz\n'
    )
  if pts_colors is not None:
    f.write(
      'property uchar red\n'
      'property uchar green\n'
      'property uchar blue\n'
    )
  if texture_uv is not None:
    f.write(
      'property float texture_u\n'
      'property float texture_v\n'
    )
  if faces is not None:
    f.write(
      'element face ' + str(len(faces)) + '\n'
                                          'property list uchar int vertex_indices\n'
    )
  if texture_uv_face is not None:
    f.write(
      'property list uchar float texcoord\n'
    )
  f.write('end_header\n')

  format_float = "{:.4f}"
  format_2float = " ".join((format_float for _ in range(2)))
  format_3float = " ".join((format_float for _ in range(3)))
  format_int = "{:d}"
  format_3int = " ".join((format_int for _ in range(3)))

  # Save vertices.
  for pt_id, pt in enumerate(pts):
    if not np.isnan(np.sum(pt)):
      f.write(format_3float.format(*pts[pt_id].astype(float)))
      if pts_normals is not None:
        f.write(' ')
        f.write(format_3float.format(*pts_normals[pt_id].astype(float)))
      if pts_colors is not None:
        f.write(' ')
        f.write(format_3int.format(*pts_colors[pt_id].astype(int)))
      if texture_uv is not None:
        f.write(' ')
        f.write(format_2float.format(*texture_uv[pt_id].astype(float)))
      f.write('\n')

  # Save faces.
  if faces is not None:
    for face_id, face in enumerate(faces):
      line = ' '.join(map(str, map(int, [len(face)] + list(face.squeeze()))))
      if texture_uv_face is not None:
        uv = texture_uv_face[face_id]
        line += ' ' + ' '.join(
          map(str, [len(uv)] + map(float, list(uv.squeeze()))))
      f.write(line)
      f.write('\n')

  f.close()









if __name__ == '__main__':
    obj_model_dir = '../data/obj_models/real_train'  #répertoire à partir duquel chercher les fichiers .obj
    ply_dir = os.path.join(obj_model_dir,'plys')  #répertoire dans lequel enregistrer les fichiers .ply
    list_obj_files, list_file_names = getFiles(obj_model_dir, 'obj')  #list_obj_files : liste des paths des fichiers .obj trouvés à partir du path obj_model_dir
    #list_file_names : liste des noms des fichiers .obj trouvés à partir du path obj_model_dir

    if not os.path.exists(ply_dir) : #si le répertoire où stocker les fichiers .ply n'existe pas
      os.makedirs(ply_dir)  #crée le répertoire

    for i in range(len(list_obj_files)) :
    	path_file = list_obj_files[i]  #path du fichier .obj
    	file_name = list_file_names[i]  #nom du fichier .obj avec l'extension
    	path_ply = os.path.join(ply_dir, file_name.replace('.obj', '.ply') )  #nom du fichier .ply à créer avec l'extension
    	mod = load_obj(path_file)  #charge le fichier .obj au path path_file dans un dict de clefs 'pts', 'faces' et éventuellement 'normals'
    	save_ply(path_ply, mod)  #crée le fichier .ply à partir du fichier .obj de path path_file et l'enregistre au path path_ply 

