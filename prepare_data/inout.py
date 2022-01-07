# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""I/O functions."""

import misc
import os
import struct
import numpy as np
import imageio
import png
import json

import cv2

def load_im(path):
  """Charge une image depuis un path.
  Input : 
    - path : string, path d'une image à charger
  Output :
    - im : array numpy contenant l'image chargée
  """
  im = cv2.imread(path,-1)
  return im


def save_im(path, im, jpg_quality=95):
  """Saves an image to a file.

  :param path: Path to the output image file.
  :param im: ndarray with the image to save.
  :param jpg_quality: Quality of the saved image (applies only to JPEG).
  """
  ext = os.path.splitext(path)[1][1:]
  if ext.lower() in ['jpg', 'jpeg']:
    imageio.imwrite(path, im, quality=jpg_quality)
  else:
    imageio.imwrite(path, im)


def load_depth(path):
  """Loads a depth image from a file.

  :param path: Path to the depth image file to load.
  :return: ndarray with the loaded depth image.
  """
  d = imageio.imread(path)
  return d.astype(np.float32)


def save_depth(path, im):
  """Saves a depth image (16-bit) to a PNG file.

  :param path: Path to the output depth image file.
  :param im: ndarray with the depth image to save.
  """
  if path.split('.')[-1].lower() != 'png':
    raise ValueError('Only PNG format is currently supported.')

  im_uint16 = np.round(im).astype(np.uint16)

  # PyPNG library can save 16-bit PNG and is faster than imageio.imwrite().
  w_depth = png.Writer(im.shape[1], im.shape[0], greyscale=True, bitdepth=16)
  with open(path, 'wb') as f:
    w_depth.write(f, np.reshape(im_uint16, (-1, im.shape[1])))


def load_json(path, keys_to_int=False):
  """Loads content of a JSON file.

  :param path: Path to the JSON file.
  :return: Content of the loaded JSON file.
  """
  # Keys to integers.
  def convert_keys_to_int(x):
    return {int(k) if k.lstrip('-').isdigit() else k: v for k, v in x.items()}

  with open(path, 'r') as f:
    if keys_to_int:
      content = json.load(f, object_hook=lambda x: convert_keys_to_int(x))
    else:
      content = json.load(f)

  return content


def save_json(path, content):
  """Saves the provided content to a JSON file.

  :param path: Path to the output JSON file.
  :param content: Dictionary/list to save.
  """
  with open(path, 'w') as f:

    if isinstance(content, dict):
      f.write('{\n')
      content_sorted = sorted(content.items(), key=lambda x: x[0])
      for elem_id, (k, v) in enumerate(content_sorted):
        f.write('  \"{}\": {}'.format(k, json.dumps(v, sort_keys=True)))
        if elem_id != len(content) - 1:
          f.write(',')
        f.write('\n')
      f.write('}')

    elif isinstance(content, list):
      f.write('[\n')
      for elem_id, elem in enumerate(content):
        f.write('  {}'.format(json.dumps(elem, sort_keys=True)))
        if elem_id != len(content) - 1:
          f.write(',')
        f.write('\n')
      f.write(']')

    else:
      json.dump(content, f, sort_keys=True)


def load_cam_params(path):
  """Loads camera parameters from a JSON file.

  :param path: Path to the JSON file.
  :return: Dictionary with the following items:
   - 'im_size': (width, height).
   - 'K': 3x3 intrinsic camera matrix.
   - 'depth_scale': Scale factor to convert the depth images to mm (optional).
  """
  c = load_json(path)

  cam = {
    'im_size': (c['width'], c['height']),
    'K': np.array([[c['fx'], 0.0, c['cx']],
                   [0.0, c['fy'], c['cy']],
                   [0.0, 0.0, 1.0]])
  }

  if 'depth_scale' in c.keys():
    cam['depth_scale'] = float(c['depth_scale'])

  return cam


def load_scene_camera(path):
  """Loads content of a JSON file with information about the scene camera.

  See docs/bop_datasets_format.md for details.

  :param path: Path to the JSON file.
  :return: Dictionary with the loaded content.
  """
  scene_camera = load_json(path, keys_to_int=True)

  for im_id in scene_camera.keys():
    if 'cam_K' in scene_camera[im_id].keys():
      scene_camera[im_id]['cam_K'] = \
        np.array(scene_camera[im_id]['cam_K'], np.float).reshape((3, 3))
    if 'cam_R_w2c' in scene_camera[im_id].keys():
      scene_camera[im_id]['cam_R_w2c'] = \
        np.array(scene_camera[im_id]['cam_R_w2c'], np.float).reshape((3, 3))
    if 'cam_t_w2c' in scene_camera[im_id].keys():
      scene_camera[im_id]['cam_t_w2c'] = \
        np.array(scene_camera[im_id]['cam_t_w2c'], np.float).reshape((3, 1))
  return scene_camera


def save_scene_camera(path, scene_camera):
  """Saves information about the scene camera to a JSON file.

  See docs/bop_datasets_format.md for details.

  :param path: Path to the output JSON file.
  :param scene_camera: Dictionary to save to the JSON file.
  """
  for im_id in sorted(scene_camera.keys()):
    im_camera = scene_camera[im_id]
    if 'cam_K' in im_camera.keys():
      im_camera['cam_K'] = im_camera['cam_K'].flatten().tolist()
    if 'cam_R_w2c' in im_camera.keys():
      im_camera['cam_R_w2c'] = im_camera['cam_R_w2c'].flatten().tolist()
    if 'cam_t_w2c' in im_camera.keys():
      im_camera['cam_t_w2c'] = im_camera['cam_t_w2c'].flatten().tolist()
  save_json(path, scene_camera)


def load_scene_gt(path):
  """Loads content of a JSON file with ground-truth annotations.

  See docs/bop_datasets_format.md for details.

  :param path: Path to the JSON file.
  :return: Dictionary with the loaded content.
  """
  scene_gt = load_json(path, keys_to_int=True)

  for im_id, im_gt in scene_gt.items():
    for gt in im_gt:
      if 'cam_R_m2c' in gt.keys():
        gt['cam_R_m2c'] = np.array(gt['cam_R_m2c'], np.float).reshape((3, 3))
      if 'cam_t_m2c' in gt.keys():
        gt['cam_t_m2c'] = np.array(gt['cam_t_m2c'], np.float).reshape((3, 1))
  return scene_gt


def save_scene_gt(path, scene_gt):
  """Saves ground-truth annotations to a JSON file.

  See docs/bop_datasets_format.md for details.

  :param path: Path to the output JSON file.
  :param scene_gt: Dictionary to save to the JSON file.
  """
  for im_id in sorted(scene_gt.keys()):
    im_gts = scene_gt[im_id]
    for gt in im_gts:
      if 'cam_R_m2c' in gt.keys():
        gt['cam_R_m2c'] = gt['cam_R_m2c'].flatten().tolist()
      if 'cam_t_m2c' in gt.keys():
        gt['cam_t_m2c'] = gt['cam_t_m2c'].flatten().tolist()
      if 'obj_bb' in gt.keys():
        gt['obj_bb'] = [int(x) for x in gt['obj_bb']]
  save_json(path, scene_gt)


def load_bop_results(path, version='bop19'):
  """Loads 6D object pose estimates from a file.

  :param path: Path to a file with pose estimates.
  :param version: Version of the results.
  :return: List of loaded poses.
  """
  results = []

  # See docs/bop_challenge_2019.md for details.
  if version == 'bop19':
    header = 'scene_id,im_id,obj_id,score,R,t,time'
    with open(path, 'r') as f:
      line_id = 0
      for line in f:
        line_id += 1
        if line_id == 1 and header in line:
          continue
        else:
          elems = line.split(',')
          if len(elems) != 7:
            raise ValueError(
              'A line does not have 7 comma-sep. elements: {}'.format(line))

          result = {
            'scene_id': int(elems[0]),
            'im_id': int(elems[1]),
            'obj_id': int(elems[2]),
            'score': float(elems[3]),
            'R': np.array(
              list(map(float, elems[4].split())), np.float).reshape((3, 3)),
            't': np.array(
              list(map(float, elems[5].split())), np.float).reshape((3, 1)),
            'time': float(elems[6])
          }

          results.append(result)
  else:
    raise ValueError('Unknown version of BOP results.')

  return results


def save_bop_results(path, results, version='bop19'):
  """Saves 6D object pose estimates to a file.

  :param path: Path to the output file.
  :param results: Dictionary with pose estimates.
  :param version: Version of the results.
  """
  # See docs/bop_challenge_2019.md for details.
  if version == 'bop19':
    lines = ['scene_id,im_id,obj_id,score,R,t,time']
    for res in results:
      if 'time' in res:
        run_time = res['time']
      else:
        run_time = -1

      lines.append('{scene_id},{im_id},{obj_id},{score},{R},{t},{time}'.format(
        scene_id=res['scene_id'],
        im_id=res['im_id'],
        obj_id=res['obj_id'],
        score=res['score'],
        R=' '.join(map(str, res['R'].flatten().tolist())),
        t=' '.join(map(str, res['t'].flatten().tolist())),
        time=run_time))

    with open(path, 'w') as f:
      f.write('\n'.join(lines))

  else:
    raise ValueError('Unknown version of BOP results.')


def check_bop_results(path, version='bop19'):
  """Checks if the format of BOP results is correct.

  :param result_filenames: Path to a file with pose estimates.
  :param version: Version of the results.
  :return: True if the format is correct, False if it is not correct.
  """
  check_passed = True
  check_msg = 'OK'
  try:
    results = load_bop_results(path, version)

    if version == 'bop19':
      # Check if the time for all estimates from the same image are the same.
      times = {}
      for result in results:
        result_key = '{:06d}_{:06d}'.format(result['scene_id'], result['im_id'])
        if result_key in times:
          if abs(times[result_key] - result['time']) > 0.001:
            check_passed = False
            check_msg = \
              'The running time for scene {} and image {} is not the same for' \
              ' all estimates.'.format(result['scene_id'], result['im_id'])
            misc.log(check_msg)
            break
        else:
          times[result_key] = result['time']

  except Exception as e:
    check_passed = False
    check_msg = 'Error when loading BOP results: {}'.format(e)
    misc.log(check_msg)

  return check_passed, check_msg


def load_ply(path):
  """Charge le maillage 3D d'un modèle 3D d'objet à partir d'un fichier .ply
  Input : 
      - path : string, path d'un fichier .ply
  Output : 
      - model : dict, de clés :
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
  f = open(path, 'rb')  #ouvre le fichier .ply

  
  n_pts = 0  #nombre de vertices dans le fichier .ply
  n_faces = 0  #nombre de faces polygonales dans le fichier .ply
  face_n_corners = 3  #nombre de sommets des faces/polygones (seules les faces triangulaires sont supportées)
  pt_props = []  #liste des propriétés des vertices introduites dans le header
  face_props = []  #liste des propriétés des faces polygonales introduites dans le header
  is_binary = False  #les fichiers .ply peuvent être en format binaire ou ASCII, ici ASCII
  header_vertex_section = False  #booléen qui sert pour la lecture du header à indiquer si la déclaration des propriétés en cours concerne les vertices
  header_face_section = False  #booléen qui sert pour la lecture du header à indiquer si la déclaration des propriétés en cours concerne les faces polygonales
  texture_file = None  #contient le path relatif au fichier .ply d'un éventuel fichier texture associé au modèle

  #lit le header du fichier
  while True:

    # Strip the newline character(s).
    line = f.readline().decode('utf8').rstrip('\n').rstrip('\r')

    if line.startswith('comment TextureFile'):  #si la ligne correspond au path d'un fichier de texture attaché au modèle
      texture_file = line.split()[-1]  #path relatif au fichier .ply du fichier texture

    elif line.startswith('element vertex'):  #si les éléments introduits par le mot-clef 'element' sont les vertices
      n_pts = int(line.split(' ')[-1])  #assigne le nombre de vertex du modèle dans le fichier .ply
      header_vertex_section = True  #indique que les propriétés associées à l'élément en cours de déclaration seront des propriétés des éléments vertex
      header_face_section = False  #indique que les propriétés associées à l'élément en cours de déclaration ne seront pas des propriétés des éléments
      #faces polygonales

    elif line.startswith('element face'):  #si les éléments introduits par le mot-clef 'element' sont les faces polygonales
      n_faces = int(line.split(' ')[-1])  #nombre de faces polygonales qui composent le modèle
      header_vertex_section = False  #indique que les propriétés associées à l'élément en cours de déclaration ne seront des propriétés des éléments vertex
      header_face_section = True #indique que les propriétés associées à l'élément en cours de déclaration seront des propriétés des éléments faces
      #polygonales


    elif line.startswith('element'):  #si la déclaration d'élément en cours concerne les éléments autres que faces polygonales ou vertices
      header_vertex_section = False  #indique que les propriétés associées à l'élément en cours de déclaration ne seront des propriétés des éléments vertex
      header_face_section = False  #indique que les propriétés associées à l'élément en cours de déclaration ne seront pas des propriétés des éléments
      #faces polygonales

    elif line.startswith('property') and header_vertex_section:  #si la ligne décrit une propriété associée aux éléments vertices
      pt_props.append((line.split()[-1], line.split()[-2]))  #ajoute à la liste des propriétés des vertices un doublet (nom, type) qui définit la
      #propriété

    elif line.startswith('property list') and header_face_section:  #si la ligne décrit une propriété liste associée aux éléments faces polygonales
      elems = line.split()
      if elems[-1] == 'vertex_indices' or elems[-1] == 'vertex_index':  #si la propriété définie par une liste est 'vertex_indices' ou 'vertex_index'
        face_props.append(('n_corners', elems[2]))  #doublet qui indique le type d'entier utilisé pour chaque élément pour décrire la longueur de la liste
        #associée à la propriété pour chaque élément face
        for i in range(face_n_corners):  #parcourt le nombre de sommets de chaque face polygonale
          face_props.append(('ind_' + str(i), elems[3]))  #ajoute à la liste des propriétés des faces un doublet (numéro, type) qui indique le type
          #des éléments de chaque élément de la liste associée à la propriété pour chaque élément face
      
      elif elems[-1] == 'texcoord':
        # (name of the property, data type)
        face_props.append(('texcoord', elems[2]))
        for i in range(face_n_corners * 2):
          face_props.append(('texcoord_ind_' + str(i), elems[3]))
      else:
        misc.log('Warning: Not supported face property: ' + elems[-1])
    
    elif line.startswith('format'):  #ligne qui décrit le format du fichier
      if 'binary' in line:  #si le format de données du fichier .ply est binaire
        is_binary = True  #indique que le format de données du fichier .ply est binaire

    elif line.startswith('end_header'):  #ligne qui indique la fin du header
      break  #sort du traitement du header

  # Prepare data structures.
  model = {}
  if texture_file is not None:  #si un fichier texture est attaché au modèle
    model['texture_file'] = texture_file  #path relatif au fichier .ply du fichier texture associé au modèle

  model['pts'] = np.zeros((n_pts, 3), np.float)  #array numpy 2D de floats avec en lignes les vertices et en colonnes les coordonnées x,y,z
  if n_faces > 0:
    model['faces'] = np.zeros((n_faces, face_n_corners), np.float)  #array numpy 2D de floats avec en lignes les faces et les 3 sommets de chaque face en 
    #colonnes

  pt_props_names = [p[0] for p in pt_props]   #liste des noms des propriétés des vertices
  face_props_names = [p[0] for p in face_props]  #liste des noms des propriétés des faces

  is_normal = False   #indique si les normales aux faces sont parmi les propriétés des vertices
  if {'nx', 'ny', 'nz'}.issubset(set(pt_props_names)):  #si les normales aux faces sont parmi les propriétés des vertices
    is_normal = True
    model['normals'] = np.zeros((n_pts, 3), np.float)  #array numpy 2D de floats avec en lignes les vertices et en colonnes les coordonnées du  vecteur normale aux 
    #faces associé au vertex

  is_color = False  #indique si les couleurs font partie des propriétés associées aux vertices
  if {'red', 'green', 'blue'}.issubset(set(pt_props_names)):  #si les couleurs font partie des propriétés associées aux vertices
    is_color = True
    model['colors'] = np.zeros((n_pts, 3), np.float)  #array numpy 2D de floats avec en lignes les vertices et en colonnes les valeurs des pixels RGB associées aux
    #vertices

  is_texture_pt = False  #indique si les vertices ont des propriétés de texture
  if {'texture_u', 'texture_v'}.issubset(set(pt_props_names)):  #si les vertices ont commme propriété 'texture_u' et 'texture_v'
    is_texture_pt = True  #indique que les vertices ont des propriétés de texture
    model['texture_uv'] = np.zeros((n_pts, 2), np.float)  #array numpy 2D (n,2) où n nombre de vertices.

  is_texture_face = False  #indique si les faces ont des propriétés de texture
  if {'texcoord'}.issubset(set(face_props_names)):  #si les faces ont commme propriété 'texture_u' et 'texture_v'
    is_texture_face = True  #indique que les faces ont des propriétés de texture
    model['texture_uv_face'] = np.zeros((n_faces, 6), np.float)   #array numpy 2D (m,6) où m nombre de faces.

  # Formats for the binary case.
  formats = {
    'float': ('f', 4),
    'double': ('d', 8),
    'int': ('i', 4),
    'uchar': ('B', 1)
  }


  #charge les vertices
  for pt_id in range(n_pts):  #parcourt les lignes du fichier .ply qui contiennent les vertices
    prop_vals = {}  #dictionnaire qui contient les valeurs des propriétés du vertex de la ligne en cours 
    load_props = ['x', 'y', 'z', 'nx', 'ny', 'nz',
                  'red', 'green', 'blue', 'texture_u', 'texture_v']  #liste des propriétés à charger pour les vertices
    
    if is_binary:  #si le fichier .ply est au format binaire 
      for prop in pt_props:
        format = formats[prop[1]]
        read_data = f.read(format[1])
        val = struct.unpack(format[0], read_data)[0]
        if prop[0] in load_props:
          prop_vals[prop[0]] = val

    else:  #si le fichier .ply n'est pas au format binaire
      elems = f.readline().decode('utf8').rstrip('\n').rstrip('\r').split()
      for prop_id, prop in enumerate(pt_props):  #parcourt la liste des propriétés déclarées pour les vertices 
        if prop[0] in load_props:  #vérifie que la propriété déclarée est bien dans la liste à charger
          prop_vals[prop[0]] = elems[prop_id]  #ajoute la propriété et sa valeur au dictionnaire des propriétés du vertex en cours 

    #enregistre les coordonnées du vertex en cours dans l'array 'pts' du dictionnaire model 
    model['pts'][pt_id, 0] = float(prop_vals['x'])
    model['pts'][pt_id, 1] = float(prop_vals['y'])
    model['pts'][pt_id, 2] = float(prop_vals['z'])

    #enregistre la normale aux faces du vertex en cours dans l'array 'normals' du dictionnaire model
    if is_normal:
      model['normals'][pt_id, 0] = float(prop_vals['nx'])
      model['normals'][pt_id, 1] = float(prop_vals['ny'])
      model['normals'][pt_id, 2] = float(prop_vals['nz'])

    #enregistre le triplet RGB associé au vertex dans l'array 'colors' du dictionnaire model
    if is_color:
      model['colors'][pt_id, 0] = float(prop_vals['red'])
      model['colors'][pt_id, 1] = float(prop_vals['green'])
      model['colors'][pt_id, 2] = float(prop_vals['blue'])

    if is_texture_pt:  #si les vertices ont des propriétés de textures
      model['texture_uv'][pt_id, 0] = float(prop_vals['texture_u'])  #ajoute les propriétés des vertices 'texture_u' dans la première colonne de model['texture_uv']
      model['texture_uv'][pt_id, 1] = float(prop_vals['texture_v'])  #ajoute les propriétés des vertices 'texture_v' dans la deuxième colonne de model['texture_uv']

  #charge les faces polygonales
  for face_id in range(n_faces):  #parcourt les lignes du fichier .ply qui contiennent les faces polygonales
    prop_vals = {}  #dictionnaire qui contient les valeurs des propriétés de la face polygonale de la ligne en cours 
    
    if is_binary:  #si le fichier .ply est au format binaire 
      for prop in face_props:
        format = formats[prop[1]]
        val = struct.unpack(format[0], f.read(format[1]))[0]
        if prop[0] == 'n_corners':
          if val != face_n_corners:
            raise ValueError('Only triangular faces are supported.')
        elif prop[0] == 'texcoord':
          if val != face_n_corners * 2:
            raise ValueError('Wrong number of UV face coordinates.')
        else:
          prop_vals[prop[0]] = val
    
    else:  #si le fichier .ply n'est pas au format binaire
      elems = f.readline().decode('utf8').rstrip('\n').rstrip('\r').split()

      for prop_id, prop in enumerate(face_props):  #parcourt la liste des propriétés déclarées pour les faces polygonales 
        if prop[0] == 'n_corners':  #traite le cas où le nombre de sommets de la face polygonale en cours n'est pas égal à 3
          if int(elems[prop_id]) != face_n_corners:
            raise ValueError('Only triangular faces are supported.')
        elif prop[0] == 'texcoord':
          if int(elems[prop_id]) != face_n_corners * 2:
            raise ValueError('Wrong number of UV face coordinates.')
        else:  #si le nombre de sommets de la face polygonale en cours est bien égal à 3
          prop_vals[prop[0]] = elems[prop_id]  #ajoute la propriété et sa valeur au dictionnaire des propriétés de la face polygonale en cours

    #enregistre les indices des vertices/sommets qui composent la face polygonale en cours dans l'array 'faces' du dictionnaire model
    model['faces'][face_id, 0] = int(prop_vals['ind_0'])
    model['faces'][face_id, 1] = int(prop_vals['ind_1'])
    model['faces'][face_id, 2] = int(prop_vals['ind_2'])

    if is_texture_face:  #si les faces ont des propriétés de texture
      for i in range(6):
        model['texture_uv_face'][face_id, i] = float(prop_vals['texcoord_ind_{}'.format(i)])  #ajoute la propriété de texture de la face à model['texture_uv_face']

  f.close()  #ferme le fichier .ply

  return model


def save_ply(path, model, extra_header_comments=None):
  """Saves a 3D mesh model to a PLY file.

  :param path: Path to a PLY file.
  :param model: 3D model given by a dictionary with items:
   - 'pts' (nx3 ndarray)
   - 'normals' (nx3 ndarray, optional)
   - 'colors' (nx3 ndarray, optional)
   - 'faces' (mx3 ndarray, optional)
   - 'texture_uv' (nx2 ndarray, optional)
   - 'texture_uv_face' (mx6 ndarray, optional)
   - 'texture_file' (string, optional)
  :param extra_header_comments: Extra header comment (optional).
  """
  pts = model['pts']
  pts_colors = model['colors'] if 'colors' in model.keys() else None
  pts_normals = model['normals'] if 'normals' in model.keys() else None
  faces = model['faces'] if 'faces' in model.keys() else None
  texture_uv = model[
    'texture_uv'] if 'texture_uv' in model.keys() else None
  texture_uv_face = model[
    'texture_uv_face'] if 'texture_uv_face' in model.keys() else None
  texture_file = model[
    'texture_file'] if 'texture_file' in model.keys() else None

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
