# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""A Python based renderer."""

import os
import numpy as np
from glumpy import app, gloo, gl

import inout
import misc
import renderer

# Set glumpy logging level.
from glumpy.log import log
import logging
log.setLevel(logging.WARNING)  # Options: ERROR, WARNING, DEBUG, INFO.

# Set backend (http://glumpy.readthedocs.io/en/latest/api/app-backends.html).
# app.use('glfw')  # Options: 'glfw', 'qt5', 'pyside', 'pyglet'.




# RGB vertex shader.
_rgb_vertex_code = """
uniform mat4 u_mv;
uniform mat4 u_nm;
uniform mat4 u_mvp;
uniform vec3 u_light_eye_pos;

attribute vec3 a_position;
attribute vec3 a_normal;
attribute vec3 a_color;
attribute vec2 a_texcoord;

varying vec3 v_color;
varying vec2 v_texcoord;
varying vec3 v_eye_pos;
varying vec3 v_L;
varying vec3 v_normal;

void main() {
    gl_Position = u_mvp * vec4(a_position, 1.0);
    v_color = a_color;
    v_texcoord = a_texcoord;
    
    // The following points/vectors are expressed in the eye coordinates.
    v_eye_pos = (u_mv * vec4(a_position, 1.0)).xyz; // Vertex.
    v_L = normalize(u_light_eye_pos - v_eye_pos); // Vector to the light.
    v_normal = normalize(u_nm * vec4(a_normal, 1.0)).xyz; // Normal vector.
}
"""

# RGB fragment shader - flat shading.
_rgb_fragment_flat_code = """
uniform float u_light_ambient_w;
uniform sampler2D u_texture;
uniform int u_use_texture;

varying vec3 v_color;
varying vec2 v_texcoord;
varying vec3 v_eye_pos;
varying vec3 v_L;

void main() {
    // Face normal in eye coords.
    vec3 f_normal = normalize(cross(dFdx(v_eye_pos), dFdy(v_eye_pos)));

    float light_diffuse_w = max(dot(normalize(v_L), normalize(f_normal)), 0.0);
    float light_w = u_light_ambient_w + light_diffuse_w;
    if(light_w > 1.0) light_w = 1.0;

    if(bool(u_use_texture)) {
        gl_FragColor = vec4(light_w * texture2D(u_texture, v_texcoord));
    }
    else {
        gl_FragColor = vec4(light_w * v_color, 1.0);
    }
}
"""

# RGB fragment shader - Phong shading.
_rgb_fragment_phong_code = """
uniform float u_light_ambient_w;
uniform sampler2D u_texture;
uniform int u_use_texture;

varying vec3 v_color;
varying vec2 v_texcoord;
varying vec3 v_eye_pos;
varying vec3 v_L;
varying vec3 v_normal;

void main() {
    float light_diffuse_w = max(dot(normalize(v_L), normalize(v_normal)), 0.0);
    float light_w = u_light_ambient_w + light_diffuse_w;
    if(light_w > 1.0) light_w = 1.0;

    if(bool(u_use_texture)) {
        gl_FragColor = vec4(light_w * texture2D(u_texture, v_texcoord));
    }
    else {
        gl_FragColor = vec4(light_w * v_color, 1.0);
    }
}
"""

# Depth vertex shader.
# Ref: https://github.com/julienr/vertex_visibility/blob/master/depth.py
#
# Getting the depth from the depth buffer in OpenGL is doable, see here:
#   http://web.archive.org/web/20130416194336/http://olivers.posterous.com/linear-depth-in-glsl-for-real
#   http://web.archive.org/web/20130426093607/http://www.songho.ca/opengl/gl_projectionmatrix.html
#   http://stackoverflow.com/a/6657284/116067
# but it is difficult to achieve high precision, as explained in this article:
# http://dev.theomader.com/depth-precision/
#
# Once the vertex is in the view coordinates (view * model * v), its depth is
# simply the Z axis. Hence, instead of reading from the depth buffer and undoing
# the projection matrix, we store the Z coord of each vertex in the color
# buffer. OpenGL allows for float32 color buffer components.
_depth_vertex_code = """
uniform mat4 u_mv;
uniform mat4 u_mvp;
attribute vec3 a_position;
attribute vec3 a_color;
varying float v_eye_depth;

void main() {
    gl_Position = u_mvp * vec4(a_position, 1.0);
    vec3 v_eye_pos = (u_mv * vec4(a_position, 1.0)).xyz; // In eye coords.

    // OpenGL Z axis goes out of the screen, so depths are negative
    v_eye_depth = -v_eye_pos.z;
}
"""

# Depth fragment shader.
_depth_fragment_code = """
varying float v_eye_depth;

void main() {
    gl_FragColor = vec4(v_eye_depth, 0.0, 0.0, 1.0);
}
"""


# Functions to calculate transformation matrices.
# Note that OpenGL expects the matrices to be saved column-wise.
# (Ref: http://www.songho.ca/opengl/gl_transform.html)


def _calc_model_view(model, view):
  """Calculates the model-view matrix (from object space to eye space).

    Inputs : 
    - model : array numpy (4,4), matrice diagonale (4,4) de floats. Contient une matrice de rotation, le vecteur de translation et une derni??re ligne de 
              binaires [0,0,0,1]. Model matrix (from object space to world space) 
    - view : array numpy (4,4). View matrix (from world space to eye space) dans le rep??re cam??ra utilis?? par OpenGL.

  Outputs : 
    - array numpy (4,4), 'model-view' matrice obtenue par multiplication model x view, contient une rotation et une translation (from object space to eye space)
  """
  return np.dot(model, view)


def _calc_model_view_proj(model, view, proj):
  """Calculates the model-view-projection matrix.

  Inputs : 
    - model : array numpy (4,4), matrice diagonale (4,4) de floats. Contient une matrice de rotation, le vecteur de translation et une derni??re ligne de 
              binaires [0,0,0,1]. Model matrix (from object space to world space) 
    - view : array numpy (4,4). View matrix (from world space to eye space) dans le rep??re cam??ra utilis?? par OpenGL.
    - proj : array numpy (4,4), contient la matrice de projection OpenGL calcul??e ?? partir de la matrice intrins??que de Hartley-Zisserman

  Outputs : 
    - array numpy (4,4), model-view-projection matrix.
  """
  return np.dot(np.dot(model, view), proj)


def _calc_normal_matrix(model, view):
  """Calcule la 'matrice normale' qui permet de transformer un vecteur normal dans l' "object space" en un vecteur normal dans l' "eye space"

  Ref: http://www.songho.ca/opengl/gl_normaltransform.html

  Inputs : 
    - model : array numpy (4,4), matrice diagonale (4,4) de floats. Contient une matrice de rotation, le vecteur de translation et une derni??re ligne de 
              binaires [0,0,0,1]. Model matrix (from object space to world space) 
    - view : array numpy (4,4). View matrix (from world space to eye space) dans le rep??re cam??ra utilis?? par OpenGL.

  Outputs : 
    - array numpy (4,4), 'matrice normale' qui permet de transformer un vecteur normal dans l' "object space" en un vecteur normal dans l' "eye space"
  """
  return np.linalg.inv(np.dot(model, view)).T  #transpos??e de l'inverse de la matrice qui permet de passer de l'object space ?? l'eye space (calcul??e par _calc_model_view)


def _calc_calib_proj(K, x0, y0, w, h, nc, fc, window_coords='y_down'):
  """Conversion de la matrice intrins??que de Hartley-Zisserman ?? la matrice de projection OpenGL

  Ref:
  1) https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL
  2) https://github.com/strawlab/opengl-hz/blob/master/src/calib_test_utils.py

  Inputs : 
    - K : array numpy (3,3), param??tres intrins??ques de la cam??ra
    - x0 : float, coordonn??e selon x de l'origine du rep??re de la cam??ra (typiquement 0)
    - y0 : float, coordonn??e selon y de l'origine du rep??re de la cam??ra (typiquement 0)
    - w : int, largeur de l'image
    - h : int, hauteur de l'image
    - nc : Near clipping plane.
    - fc : Far clipping plane.
    - window_coords : string, 'y_up' ou 'y_down', ??

  Output : 
    - proj : array numpy (4,4), matrice de projection OpenGL.
  """
  depth = float(fc - nc)  #profondeur de la bbox 3D
  q = -(fc + nc) / depth  #??
  qn = -2 * (fc * nc) / depth  #??

  # Draw our images upside down, so that all the pixel-based coordinate systems are the same.
  if window_coords == 'y_up':
    proj = np.array([
      [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
      [0, -2 * K[1, 1] / h, (-2 * K[1, 2] + h + 2 * y0) / h, 0],
      [0, 0, q, qn],  # Sets near and far planes (glPerspective).
      [0, 0, -1, 0]
    ])

  # Draw the images upright and modify the projection matrix so that OpenGL will generate window coords that compensate for the flipped image coords.
  else:
    assert window_coords == 'y_down'
    proj = np.array([
      [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
      [0, 2 * K[1, 1] / h, (2 * K[1, 2] - h + 2 * y0) / h, 0],
      [0, 0, q, qn],  # Sets near and far planes (glPerspective).
      [0, 0, -1, 0]
    ])
  return proj.T


class RendererPython(renderer.Renderer):
  """A Python based renderer."""

  def __init__(self, width, height, mode='rgb+depth', shading='phong', bg_color=(0.0, 0.0, 0.0, 0.0)):
    """Constructeur.

    Input : 
      - width : int, largeur de l'image repr??sent??e/rendered
      - height: int, hauteur de l'image repr??sent??e/rendered
      - mode : string, mode de rendering ('rgb+depth', 'rgb', 'depth'). Par d??faut 'rgb+depth'
      - shading : string, type d'ombrage/shading ('flat', 'phong'). Par d??gaut 'phong'
      - bg_color: quadruplet de floats, couleur + channel alpha du background (R, G, B, A). Par d??faut transparent et noir.
    """
    super(RendererPython, self).__init__(width, height)  #appel au constructeur de la classe Renderer du fichier 'renderer.py'

    self.mode = mode  #string, mode de rendering
    self.shading = shading  #string, mode d'ombrage
    self.bg_color = bg_color  #quadruplet de floats, couleur du background de l'image + channel alpha de transparence

    #Indicateurs pour utiliser l'image RGB et/ou la map de profondeur
    self.render_rgb = self.mode in ['rgb', 'rgb+depth']  #bool??en, indique si on utilise l'image RGB
    self.render_depth = self.mode in ['depth', 'rgb+depth']  #bool??en, indique si on utilise la map de profondeur

    #Dicts pour stocker les mod??les d'objets et leurs infos
    self.models = {}  #en clefs les ids des mod??les des objets, en valeurs les dicts retourn??s par load_ply lorsqu'on charge le fichier .ply des mod??les d'objets 
    self.model_bbox_corners = {}  #en clefs les ids des mod??les des objets, en valeurs les coins de la bounding box 3D du mod??le de l'objet, array numpy 2D (8,3)
    self.model_textures = {}

    #Images repr??sent??e/rendered
    self.rgb = None
    self.depth = None

    # Fen??tre de rendering.
    # self.window = app.Window(visible=False)
    config = app.configuration.Configuration()  #modifi??
    self.window = app.Window(visible=False, width = self.width, height = self.height, config = config)  #modifi??, ne fonctionne pas ligne du dessus. Fen??tre pour afficher les images (?)

    
    self.vertex_buffers = {}  #Buffer utilis?? pour chaque objet pour stocker les vertices 
    self.index_buffers = {}  #Buffer utilis?? pour chaque objet pour stocker les indices

    self.rgb_programs = {}  #Programme OpenGL utilis??s pour chaque objet pour repr??senter/render les images rgb
    self.depth_programs = {}  #Programme OpenGL utilis??s pour chaque objet pour repr??senter/render les cartes de profondeur

    
    rgb_buf = np.zeros((self.height, self.width, 4), np.float32).view(gloo.TextureFloat2D)  #objet buffer de couleur (Color buffer). 4 channels pour RGB + channel alpha
    depth_buf = np.zeros((self.height, self.width), np.float32).view(gloo.DepthTexture)  #objet buffer de carte de profondeur (Depth buffer)
    self.fbo = gloo.FrameBuffer(color=rgb_buf, depth=depth_buf)  #L'objet Framebuffer, collection de buffers qui peuvent ??tre utilis??s comme la destination pour le rendering
    self.fbo.activate()  #active l'objet frame buffer, l'active sur GPU

  def add_object(self, obj_id, model_path, **kwargs):
    """Charge un mod??le d'objet.
    Inputs :
      - obj_id : string, identifiant de l'objet
      - model_path : string, path du fichier .ply du mod??le de l'objet
    """

    surf_color = None  #couleur du mod??le de l'objet, si pas pass??e en argument la couleur utilis??e sera celle enregistr??e avec le mod??le de l'objet
    if 'surf_color' in kwargs:  #si l'argument 'surf_colors' a ??t?? pass?? en param??tres suppl??mentaires
      surf_color = kwargs['surf_color']  #charge la couleur de l'objet pass??e en argument

    # Load the object model.

    model = inout.load_ply(model_path)  #dict, contient le mod??le de l'objet. Parmi les clefs : 'pts' : array numpy 2D (n,3), contient les coordonn??es des vertices selon
    #les axes x,y,z. 
    
    self.models[obj_id] = model  #enregistre le dict contenant le mod??le de l'objet charg??

    # Calcule la bounding box 3D de l'objet (sera utilis?? pour cr??er les plans de coupe proches et lointains)
    bb = misc.calc_3d_bbox( model['pts'][:, 0], model['pts'][:, 1], model['pts'][:, 2])  #liste ?? 6 ??l??ments, contient la bounding box 3D de l'objet, (x, y, z, w, h, d),
    # o?? (x, y, z) est le coin haut-gauche, et (w, h, d) est la largeur, hauteur, profondeur de la bounding box.

    self.model_bbox_corners[obj_id] = np.array([ [bb[0], bb[1], bb[2]],  [bb[0], bb[1], bb[2] + bb[5]],  [bb[0], bb[1] + bb[4], bb[2]], [bb[0], bb[1] + bb[4], bb[2] + bb[5]],
    [bb[0] + bb[3], bb[1], bb[2]], [bb[0] + bb[3], bb[1], bb[2] + bb[5]], [bb[0] + bb[3], bb[1] + bb[4], bb[2]], [bb[0] + bb[3], bb[1] + bb[4], bb[2] + bb[5]], ])  #array
    #numpy 2D (8,3) contient les coins de la bounding box 3D

    # Set texture/color of vertices.
    self.model_textures[obj_id] = None  #initialise la couleur/texture des vertices


    if surf_color is not None:  #si la couleur de surface (uniforme) a ??t?? pass??e dans les co-arguments de la m??thode
      colors = np.tile(list(surf_color) + [1.0], [model['pts'].shape[0], 1])  #array numpy 2D (n,m+1) o?? n nombre de vertices dans le mod??le, et m longueur de surf_color.
      #(probablement 3). Contient pour chaque vertex la couleur ?? appliquer (probablement un quadruplet RGBA, le dernier ??tant la transparence fix??e ?? 1.0, opaque) 

      texture_uv = np.zeros((model['pts'].shape[0], 2), np.float32)  #array numpy 2D (n,2) o?? n nombre de vertices du mod??le. Initialise les coordonn??es de la texture UV

    # Use the model texture.
    elif 'texture_file' in self.models[obj_id].keys():  #si le mod??le dispose d'un fichier texture associ?? ?? son fichier .ply
      model_texture_path = os.path.join(os.path.dirname(model_path), self.models[obj_id]['texture_file'])  #path du fichier texture 
      model_texture = inout.load_im(model_texture_path)  #array numpy 3D qui charge la texture 

      # Normalize the texture image.
      if model_texture.max() > 1.0:  #si les valeurs de texture sont cod??es entre 0 et 255 et non entre 0 et 1
        model_texture = model_texture.astype(np.float32) / 255.0  #ram??ne les valeurs de texture entre 0 et 1
      model_texture = np.flipud(model_texture)  #inverse l'ordre des ??l??ments de model_texture selon la premi??re dimension (pour mettre en RGB car cv2 charge en BGR?)
      self.model_textures[obj_id] = model_texture  #enregistre la texture du mod??le 


      texture_uv = model['texture_uv']  #array numpy nx2 o?? n nombre de vertices utilis??s pour d??crire l'objet. Coordonn??es UV de texture associ??es ?? chaque vertex. 
      colors = np.zeros((model['pts'].shape[0], 3), np.float32)  #array numpy 2D (n,3) o?? n nombre de vertices. Contient la couleur associ??e ?? chaque vertex

    # Use the original model color.
    elif 'colors' in model.keys():  #si la couleur des vertices est renseign??e dans le fichier .ply
      assert (model['pts'].shape[0] == model['colors'].shape[0])  #v??rifie que chaque vertex a bien une couleur associ??e
      colors = model['colors']  #array numpy (n,3), contient pour chaque vertex son triplet RGB associ??
      if colors.max() > 1.0:  #si les pixels sont cod??s entre 0 et 255
        colors /= 255.0  #ram??ne les valeurs des pixels entre 0 et 1

      texture_uv = np.zeros((model['pts'].shape[0], 2), np.float32)  #array numpy nx2 o?? n nombre de vertices. Coordonn??es UV de texture associ??es ?? chaque vertex,
      #initialis??es ?? 0

    # Set the model color to gray.
    else:   #Si aucune couleur n'est renseign??e en argument de la fonction ni dans le fichier .ply
      colors = np.ones((model['pts'].shape[0], 3), np.float32) * 0.5  ##array numpy 2D (n,3) o?? n nombre de vertices. Fixe la couleur de chaque vertex ?? gris
      texture_uv = np.zeros((model['pts'].shape[0], 2), np.float32)  #array numpy nx2 o?? n nombre de vertices. Coordonn??es UV de texture associ??es ?? chaque vertex, 
      #initialis??es ?? 0

    #cr??e les donn??es associ??es aux vertices
    if self.mode == 'depth':  #si le mode de rendering n'utilise pas d'images RGB
      vertices_type = [ ('a_position', np.float32, 3), ('a_color', np.float32, colors.shape[1]) ]  #dtype de l'array vertices
      vertices = np.array(list(zip(model['pts'], colors)), vertices_type)  #array numpy 1D ?? 1 ??l??ment qui contient un tuple contenant les deux arrays model['pts] et colors.
      #type de donn??es de l'array : vertices_type

    else:  #si le mode de rendering utilise des images RGB
      if self.shading == 'flat':  #si le mode d'ombrage est flat
        vertices_type = [ ('a_position', np.float32, 3),  ('a_color', np.float32, colors.shape[1]),  ('a_texcoord', np.float32, 2)]  #dtype de l'array vertices
        vertices = np.array(list(zip(model['pts'], colors, texture_uv)),  vertices_type)  #array numpy 1D ?? 1 ??l??ment qui contient un tuple contenant les trois arrays 
        #model['pts], colors et texture_uv. dtype : vertices_type
      
      elif self.shading == 'phong':  #si le mode d'ombrage est phong
        vertices_type = [('a_position', np.float32, 3), ('a_normal', np.float32, 3), ('a_color', np.float32, colors.shape[1]), ('a_texcoord', np.float32, 2)]  #dtype de
        #l'array vertices
        vertices = np.array(list(zip(model['pts'], model['normals'],  colors, texture_uv)), vertices_type)  #array numpy 1D ?? 1 ??l??ment qui contient un tuple contenant les
        #quatre arrays model['pts], model['normals'], colors et texture_uv. dtype : vertices_type

      elif self.shading == 'cate':  #si le mode d'ombrage est phong
        vertices_type = [('a_position', np.float32, 3)]  #dtype de l'array vertices
        vertices = np.array(list(zip(model['pts'])), vertices_type)  #array numpy 1D ?? 1 ??l??ment qui contient un tuple contenant l'array model['pts]. dtype : vertices_type
      
      else:  #si le mode d'ombrage n'est pas dans les choix pr??c??dents
        raise ValueError('Unknown shading type.')  #l??ve une erreur

    #Cr??e les buffers des vertices et des indices pour le mod??le d'objet charg??
    self.vertex_buffers[obj_id] = vertices.view(gloo.VertexBuffer)
    self.index_buffers[obj_id] = model['faces'].flatten().astype(np.uint32).view(gloo.IndexBuffer)

    #Cr??e le shader pour l'ombrage s??lectionn??
    if self.shading == 'flat':
      rgb_fragment_code = _rgb_fragment_flat_code  #associe le code en haut du fichier qui d??finit le shader dans le cas de l'ombrage flat
    elif self.shading == 'phong':
      rgb_fragment_code = _rgb_fragment_phong_code  #associe le code en haut du fichier qui d??finit le shader dans le cas de l'ombrage phong
    else:  #si le shading n'est ni phong ni flat
      rgb_fragment_code = _rgb_fragment_phong_code  #par d??faut, le shading est phong
      # raise ValueError('Unknown shading type.')

    # Prepare the RGB OpenGL program.
    rgb_program = gloo.Program(_rgb_vertex_code, rgb_fragment_code)  #objet auquel peuvent ??tre attach??s et li??s les shaders pour cr??er un programme de shader pour les
    #images RGB
    rgb_program.bind(self.vertex_buffers[obj_id])  #ajoute le buffer des vertices au programme
    
    if self.model_textures[obj_id] is not None:  #si la texture a ??t?? renseign??e
      rgb_program['u_use_texture'] = int(True)
      rgb_program['u_texture'] = self.model_textures[obj_id]
    else:
      rgb_program['u_use_texture'] = int(False)
      rgb_program['u_texture'] = np.zeros((1, 1, 4), np.float32)
    self.rgb_programs[obj_id] = rgb_program  #programme OpenGL utilis??s pour chaque objet pour repr??senter/render les images RGB

    # Prepare the depth OpenGL program.
    depth_program = gloo.Program(_depth_vertex_code,_depth_fragment_code)  #objet auquel peuvent ??tre attach??s et li??s les shaders pour cr??er un programme de shader pour les
    #cartes de profondeur
    depth_program.bind(self.vertex_buffers[obj_id])  #ajoute le buffer des vertices au programme
    self.depth_programs[obj_id] = depth_program  #programme OpenGL utilis??s pour chaque objet pour repr??senter/render les cartes de profondeur

  def remove_object(self, obj_id):
    """See base class."""
    del self.models[obj_id]
    del self.model_bbox_corners[obj_id]
    if obj_id in self.model_textures:
      del self.model_textures[obj_id]
    del self.vertex_buffers[obj_id]
    del self.index_buffers[obj_id]
    del self.rgb_programs[obj_id]
    del self.depth_programs[obj_id]

  def render_object(self, obj_id, R, t, fx, fy, cx, cy):
    """Renders an object model in the specified pose.

    Inputs :
      - obj_id : string, nom du mod??le de l'objet dans l'image. Doit matcher les noms des fichiers .ply qui d??finissent les mod??les des objets (clefs de self.rgb_program
                  et self.depth_program)?
      - R : array numpy 2D (3,3), matrice de rotation de l'objet dans l'image
      - t : array numpy 2D (3,1), vecteur de translation de l'objet dans l'image
      - fx : floar, premier ??l??ment de la diagonale de la matrice des param??tres intrins??ques de la cam??ra. Distance focale selon l'axe x
      - fy : float, second ??l??ment de la diagonale de la matrice des param??tres intrins??ques de la cam??ra. Distance focale selon l'axe y
      - cx : float, 1er ??l??ment de la derni??re colonne de la matrice des param??tres intrins??ques de la cam??ra. Coordonn??es selon x du point principal
      - cy : float, 2??me ??l??ment de la derni??re colonne de la matrice des param??tres intrins??ques de la cam??ra. Coordonn??es selon y du point principal
      Le point principal est l'origine du rep??re de l'image (coin bas gauche). Ses coordonn??es sont exprim??es dans le rep??re (plan) de la cam??ra.

    Output :
      - dict de clefs :
        -> 'rgb' : rgb : array numpy (h,w,3) o?? h,w dimensions de l'image. Contient les pixels (dans [0,255]) de l'image avec l'objet d'id curr_obj_id rendered/repr??sent?? 
        dessus
        
        et/ou
        
        -> 'depth' : array numpy 2D (h,w) o?? h,w dimensions de l'image. Contient la carte de profondeur de l'image avec l'objet d'id obj_id 
        rendered/repr??sent?? dessus.
    """
    
    global curr_obj_id, mat_model, mat_view, mat_proj  #D??finit des variables globales pour pouvoir y acc??der dans les m??thodes draw_.. ci-dessous
    curr_obj_id = obj_id  #string, nom du mod??le de l'objet dans l'image

    
    mat_model = np.eye(4, dtype=np.float32) #matrice diagonale (4,4) de floats. Contiendra une matrice de rotation, le vecteur de translation et une derni??re ligne de 
    #binaires [0,0,0,1], cf https://www.brainvoyager.com/bv/doc/UsersGuide/CoordsAndTransforms/SpatialTransformationMatrices.html pour l'explication 
    #Model matrix (from object space to world space).

    mat_view_cv = np.eye(4, dtype=np.float32) ##matrice diagonale (4,4) de floats. View matrix (from world space to eye space; transforms also the coordinate system from 
    #OpenCV to OpenGL camera space). Contiendra une matrice de rotation, le vecteur de translation et une derni??re ligne de binaires [0,0,0,1]

    mat_view_cv[:3, :3], mat_view_cv[:3, 3] = R, t.squeeze()  #met dans la matrice mat_view_cv la rotation et la translation de l'objet dans la sc??ne 

    yz_flip = np.eye(4, dtype=np.float32)  #matrice diagonale (4,4) de floats. Contiendra une matrice de rotation, le vecteur de translation et une derni??re ligne de 
    #binaires [0,0,0,1]

    yz_flip[1, 1], yz_flip[2, 2] = -1, -1  #rotation dans yz_flip : rotation d'angle pi autour de l'axe x. Inverse les axes y et z.
    mat_view = yz_flip.dot(mat_view_cv)  #Passe la matrice mat_view dans le rep??re cam??ra utilis?? par OpenGL depuis le rep??re utilis?? OpenCV en ??changeant les axes y et z.
    #View matrix (from world space to eye space)
    
    mat_view = mat_view.T  #transpose la matrice mat_view, OpenGL attend un format de matrices 'column-wise'
    
    #calcule le plan de coupe proche et lointain de la bbox 3D

    bbox_corners = self.model_bbox_corners[obj_id]  #coordonn??es des coins de la bbox 3D
    bbox_corners_ht = np.concatenate( (bbox_corners, np.ones((bbox_corners.shape[0], 1))), axis=1).transpose()  #array numpy 2D (4,8) qui contient les coordonn??es 
    #des 8 coins de la bbox 3D et une ligne de 1

    bbox_corners_eye_z = mat_view_cv[2, :].reshape((1, 4)).dot(bbox_corners_ht)  #array numpy 2D (1,8), contient??
    clip_near = bbox_corners_eye_z.min()  #float, ??
    clip_far = bbox_corners_eye_z.max()  #float, ??

    # Projection matrix.
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])  #matrice des param??tres intrins??ques de la cam??ra
    mat_proj = _calc_calib_proj( K, 0, 0, self.width, self.height, clip_near, clip_far)  #array numpy (4,4), matrice de projection OpenGL.

    @self.window.event  #decorator, doc : https://en.wikipedia.org/wiki/Python_syntax_and_semantics#Decorators. ??quivalent ?? d??finir on_draw puis ?? faire 
    #on_draw = event(on_draw) (event est une m??thode qui modifie les fonctions)
   
    def on_draw(dt):
      self.window.clear()  #supprime tout ce qu'il y a sur la fen??tre actuelle
      global curr_obj_id, mat_model, mat_view, mat_proj  #D??finit des variables globales pour pouvoir y acc??der dans les m??thodes draw_.. ci-dessous

      
      if self.render_rgb:  #si l'image rgb est utilis??e dans le rendering
        self.rgb = self._draw_rgb(curr_obj_id, mat_model, mat_view, mat_proj)  #rgb : array numpy (h,w,3) o?? h,w dimensions de l'image. Contient les pixels (dans [0,255])
        #de l'image avec l'objet d'id curr_obj_id rendered/repr??sent?? dessus
      
      if self.render_depth:  #si l'image rgb est utilis??e dans le rendering
        self.depth = self._draw_depth(curr_obj_id, mat_model, mat_view, mat_proj)  #array numpy 2D (h,w) o?? h,w dimensions de l'image. Contient la carte de profondeur de 
        #l'image avec l'objet d'id obj_id rendered/repr??sent?? dessus.

    
    app.run(framecount=0)  #The on_draw function is called framecount+1 times.  gives control to the glumpy application loop that will respond to application events such as 
    #the mouse and the keyboard.

    #enregistre l'image RGB/la carte de profondeur avec l'objet d'id obj_id rendered dessus dans un dict
    if self.mode == 'rgb':
      return {'rgb': self.rgb}
    elif self.mode == 'depth':
      return {'depth': self.depth}
    elif self.mode == 'rgb+depth':
      return {'rgb': self.rgb, 'depth': self.depth}

  def _draw_rgb(self, obj_id, mat_model, mat_view, mat_proj):
    """Effectue le rendering d'un objet d'id obj_id dans une image RGB.
    Inputs : 
      - obj_id: string, nom du mod??le de l'objet dans l'image ?? repr??senter/render
      - mat_model: array numpy 2D(4,4), matrice diagonale (4,4) de floats. Contiendra une matrice de rotation, le vecteur de translation et une derni??re ligne de 
                  binaires [0,0,0,1]. Model matrix (from object space to world space) 
      - mat_view: array numpy 2D (4,4). View matrix (from world space to eye space) dans le rep??re cam??ra utilis?? par OpenGL.
      - mat_proj: array numpy 2D (4,4), matrice de projection OpenGL.

    Output : 
      - rgb : array numpy 3D (h,w,3) o?? h,w dimensions de l'image. Contient les pixels (dans [0,255]) de l'image avec l'objet d'id obj_id rendered/repr??sent?? dessus.
    """
    # Update the OpenGL program.
    program = self.rgb_programs[obj_id]  #programme OpenGL utilis??s pour chaque objet pour repr??senter/render les images RGB  pour l'objet d'id obj_id
    program['u_light_eye_pos'] = list(self.light_cam_pos)  #liste [0, 0, 0], coordonn??es 3D d'un 'point light' dans les coordonn??es de la cam??ra
    program['u_light_ambient_w'] = self.light_ambient_weight  #float 0.5
    program['u_mv'] = _calc_model_view(mat_model, mat_view)  #array numpy (4,4), 'model-view' matrice obtenue par multiplication model x view, contient une rotation et une
    #translation (from object space to eye space)

    program['u_nm'] = _calc_normal_matrix(mat_model, mat_view)  #array numpy (4,4), 'matrice normale' qui permet de transformer un vecteur normal dans l' "object space" en 
    #un vecteur normal dans l' "eye space"

    program['u_mvp'] = _calc_model_view_proj(mat_model, mat_view, mat_proj)  #array numpy (4,4), model-view-projection matrix.

    # OpenGL setup.
    gl.glEnable(gl.GL_DEPTH_TEST)  #Effectue des comparaisons de profondeur et met ?? jour le depth buffer
    gl.glClearColor(self.bg_color[0], self.bg_color[1], self.bg_color[2], self.bg_color[3])  #specify clear values for the color buffers
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)  #clear buffers to preset values
    gl.glViewport(0, 0, self.width, self.height)  #cr??e la fen??tre d'affichage de dimensions celles de l'image    
    gl.glDisable(gl.GL_CULL_FACE)  #Keep the back-face culling disabled because of objects which do not have well-defined surface (e.g. the lamp from the lm dataset).
    program.draw(gl.GL_TRIANGLES, self.index_buffers[obj_id]) #Rendering, dessine l'objet dans la fen??tre?

    # Get the content of the FBO (frame buffer object) texture.
    rgb = np.zeros((self.height, self.width, 4), dtype=np.float32)  #array numpy (h,w,4) o?? h, w dimensions de l'image, contient les pixels RGBA del'image avec l'objet
    #rendered

    gl.glReadPixels(0, 0, self.width, self.height, gl.GL_RGBA, gl.GL_FLOAT, rgb)  #read a block of pixels from the frame buffer, met dans rgb les pixels de l'image avec
    #l'objet rendered, 3 premi??res colonnes pour RGB et 4??me pour opacit?? (channel alpha)

    rgb.shape = (self.height, self.width, 4)  #force rgb ?? ??tre de shape (h,w,4) o?? h,w dimensions de l'image
    rgb = rgb[::-1, :]  #inverse l'ordre des ??l??ments selon la premi??re dimension de rgb (pourquoi?)
    rgb = np.round(rgb[:, :, :3] * 255).astype(np.uint8)   #Met les pixels RGB dans l'intervalle [0, 255], supprime le channel alpha

    return rgb

  def _draw_depth(self, obj_id, mat_model, mat_view, mat_proj):
    """Renders a depth image.

    Inputs : 
      - obj_id: string, nom du mod??le de l'objet dans l'image ?? repr??senter/render
      - mat_model: array numpy 2D (4,4), matrice diagonale (4,4) de floats. Contiendra une matrice de rotation, le vecteur de translation et une derni??re ligne de 
                  binaires [0,0,0,1]. Model matrix (from object space to world space) 
      - mat_view: array numpy 2D (4,4). View matrix (from world space to eye space) dans le rep??re cam??ra utilis?? par OpenGL.
      - mat_proj: array numpy 2D (4,4), matrice de projection OpenGL.

    Output : 
      - depth : array numpy 2D (h,w) o?? h,w dimensions de l'image. Contient la carte de profondeur de l'image avec l'objet d'id obj_id rendered/repr??sent?? dessus.

    :return: HxW ndarray with the rendered depth image.
    """
    # Update the OpenGL program.
    program = self.depth_programs[obj_id]  #programme OpenGL utilis??s pour chaque objet pour repr??senter/render les cartes de profondeur pour l'objet d'id obj_id
    program['u_mv'] = _calc_model_view(mat_model, mat_view)  #array numpy (4,4), 'model-view' matrice obtenue par multiplication model x view, contient une rotation et une
    #translation (from object space to eye space)

    program['u_mvp'] = _calc_model_view_proj(mat_model, mat_view, mat_proj)  #array numpy (4,4), model-view-projection matrix.

    # OpenGL setup.
    gl.glEnable(gl.GL_DEPTH_TEST)  #Effectue des comparaisons de profondeur et met ?? jour le depth buffer
    gl.glClearColor(0.0, 0.0, 0.0, 0.0)  #specify clear values for the color buffers
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)  #clear buffers to preset values
    gl.glViewport(0, 0, self.width, self.height)  #cr??e la fen??tre d'affichage de dimensions celles de l'image    
    gl.glDisable(gl.GL_CULL_FACE)  #Keep the back-face culling disabled because of objects which do not have well-defined surface (e.g. the lamp from the lm dataset).    
    program.draw(gl.GL_TRIANGLES, self.index_buffers[obj_id])  #Rendering, dessine l'objet dans la fen??tre?

    # Get the content of the FBO texture.
    depth = np.zeros((self.height, self.width, 4), dtype=np.float32)   #array numpy (h,w,4) o?? h, w dimensions de l'image, contient les pixels RGBA de l'image avec l'objet
    #rendered

    gl.glReadPixels(0, 0, self.width, self.height, gl.GL_RGBA, gl.GL_FLOAT, depth)   #read a block of pixels from the frame buffer, met dans depth les pixels de la carte 
    #de profondeur avec l'objet rendered, 3 premi??res colonnes pour RGB et 4??me pour opacit?? (channel alpha)

    depth.shape = (self.height, self.width, 4)  #force depth ?? ??tre de shape (h,w,4) o?? h,w dimensions de l'image
    depth = depth[::-1, :]   #inverse l'ordre des ??l??ments selon la premi??re dimension de depth (pourquoi?)
    depth = depth[:, :, 0]  #ne conserve que le premier channel, o?? est pr??sente la carte de profondeur

    return depth
