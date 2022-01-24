# @Time    : 12/05/2021
# @Author  : Wei Chen
# @Project : Pycharm



from __future__ import print_function

import os
import argparse
import torch.optim as optim
from torch.autograd import Variable

import torch
from Net_archs import GCN3D_segR, Rot_green, Rot_red, Point_center_res_cate
from data_loader_fsnet import load_pts_train_cate
import torch.nn as nn
import numpy as np
import time
from uti_tool import data_augment

from pyTorchChamferDistance.chamfer_distance import ChamferDistance

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=14, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)  # nombre de workers pour charger les données
parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')  #nombre d'epochs pour l'entraînement
parser.add_argument('--outf', type=str, default='models', help='output folder')  #dossier de sortie
parser.add_argument('--outclass', type=int, default=2, help='point class')  #nombre de classes (?)
parser.add_argument('--model', type=str, default='', help='model path')  #path des modèles des objets

opt = parser.parse_args()


kc = opt.outclass
num_cor = 3
num_vec = 8
nw=0 # number of cpu
localtime = (time.localtime(time.time()))   #obtient le temps actuel
year = localtime.tm_year  #année du début du training
month = localtime.tm_mon  #mois du début du training
day = localtime.tm_mday  #jour du début du training
hour = localtime.tm_hour  #heure du début du training

cats = ['bottle','bowl','can','camera','laptop','mug']  #à modifier, catégories pour NOCS Real

for cat in cats :  #à modifier, n'entraîne que sur la catégorie 'laptop' pour NOCS


    classifier_seg3D = GCN3D_segR(class_num=2, vec_num = 1,support_num= 7, neighbor_num= 10)  #module de convolution de graphe 3D
    classifier_ce = Point_center_res_cate() #module d'estimation de la translation
    classifier_Rot_red = Rot_red(F=1296, k= 6)  #module d'estimation d'un des 2 vecteurs qui représente la rotation (le rouge)
    classifier_Rot_green = Rot_green(F=1296, k=6)  #module d'estimation d'un des 2 vecteurs qui représente la rotation (le vert)


    num_classes = opt.outclass  #nombre de classes du dataset (?)

    Loss_seg3D = nn.CrossEntropyLoss()  #fonction de coût cross-entropy
    Loss_func_ce = nn.MSELoss()  #fonction de coût Mean Square Error (pour la translation)
    Loss_func_Rot1 = nn.MSELoss()  #fonction de coût Mean Square Error
    Loss_func_Rot2 = nn.MSELoss()  #fonction de coût Mean Square Error
    Loss_func_s = nn.MSELoss()  #fonction de coût Mean Square Error pour la prédiction de la taille de l'objet




    classifier_seg3D = nn.DataParallel(classifier_seg3D)  #conteneur qui contient le module classifier_seg3D et parallélise les calculs
    classifier_ce = nn.DataParallel(classifier_ce)  #conteneur qui contient le module classifier_ce et parallélise les calculs
    classifier_Rot_red = nn.DataParallel(classifier_Rot_red)  #conteneur qui contient le module classifier_Rot_red et parallélise les calculs
    classifier_Rot_green = nn.DataParallel(classifier_Rot_green)  #conteneur qui contient le module classifier_Rot_green et parallélise les calculs

    #met les modules en mode entraînement, change la valeur de l'attribut training du module en True
    classifier_seg3D = classifier_seg3D.train()
    classifier_ce = classifier_ce.train()
    classifier_Rot_red = classifier_Rot_red.train()
    classifier_Rot_green = classifier_Rot_green.train()


    #déplace modules et fonctions de coût sur GPU
    Loss_seg3D.cuda()
    Loss_func_ce.cuda()
    Loss_func_Rot1.cuda()
    Loss_func_Rot2.cuda()
    Loss_func_s.cuda()

    classifier_seg3D.cuda()
    classifier_ce.cuda()
    classifier_Rot_red.cuda()
    classifier_Rot_green.cuda()


    opt.outf = 'models/FS_Net_%s'%(cat)  #répertoire de sortie au nom de la classe en cours
    try:  #essaye de créer le répertoire de sortie
        os.makedirs(opt.outf)
    except OSError:
        pass

    sepoch  = 0  #à modifier? nombre minimal d'epochs (?)
    batch_size = 12  #à modifier? taille du batch
    lr = 0.001  #à modifier? taux d'apprentissage pour la descente du gradient
    epochs = opt.nepoch  #nombre d'epochs

     #crée l'optimizer
    optimizer = optim.Adam([{'params': classifier_seg3D.parameters()},{'params': classifier_ce.parameters()},{'params': classifier_Rot_red.parameters()},{'params': classifier_Rot_green.parameters()}], lr=lr, betas=(0.9, 0.99))


    bbxs = 0 #? rapport avec les bounding boxes? pas utilisé par la suite
    K = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])  #paramètres intrinsèques  de la caméra

    data_path = 'data/Real/train/pts'  #à modifier path où chercher les données  (root) avant '/home/wei/Documents/code/data_sets/data_NOCS/Real/train/pts/'
    dataloader = load_pts_train_cate(data_path, batch_size, K,cat, lim=1, rad=300, shuf=True, drop=True, corners=0,nw=nw)  #objet DataLoader itérable qui contient le dataset, issu de la catégorie cate (objet CateDataset)

    for epoch in range(sepoch,epochs):  #parcourt les epochs

        if epoch > 0 and epoch % (epochs // 5) == 0:  #?
            lr = lr / 4


        optimizer.param_groups[0]['lr'] = lr  #indique le learning rate utilisé dans l'optimizer Adam pour le module de convolution 3D de graphe
        optimizer.param_groups[1]['lr'] = lr * 10  #indique le learning rate utilisé dans l'optimizer Adam pour le module d'estimation de la translation
        optimizer.param_groups[2]['lr'] = lr * 20  #indique le learning rate utilisé dans l'optimizer Adam pour le module d'estimation d'un des vecteurs de la rotation (le rouge)
        optimizer.param_groups[3]['lr'] = lr * 20  #indique le learning rate utilisé dans l'optimizer Adam pour le module d'un des vecteurs de la rotation (le vert)

        for i, data in enumerate(dataloader):  #parcourt les batchs du dataset pour la catégorie cat

            points, target_, Rs, Ts, obj_id,S, imgp= data['points'], data['label'], data['R'], data['T'], data['cate_id'], data['scale'], data['dep'] 
            ptsori = points.clone()

            target_seg = target_[:, :, 0]  #masque de segmentation de l'objet 2D (ou 3D?) ground-truth

            points_ = points.numpy().copy()

            points, corners, centers, pts_recon = data_augment(points_[:, :, 0:3], Rs, Ts,num_cor, target_seg,a=15.0)   #applique l'augmentation de données au nuage de points pour créer des versions déformées

            points, target_seg, pts_recon = Variable(torch.Tensor(points)), Variable(target_seg), Variable(pts_recon)  #wrap les tenseurs ground-truth du nuage de points, du masque de segmentation et de ? dans une Variable

            points, target_seg,pts_recon = points.cuda(), target_seg.cuda(), pts_recon.cuda()   #déplace les tenseurs ground-truth du nuage de points, du masque de segmentation et de ? sur GPU

            pointsf = points[:, :, 0:3].unsqueeze(2)  #insère une dimension

            optimizer.zero_grad()  #remet les gradients de l'Optimizer à zéro
            points = pointsf.transpose(3, 1)
            points_n = pointsf.squeeze(2)  #supprime les dimensions égales à 1

            obj_idh = torch.zeros((1,1))  #tenseur des indices dans one_hot

            if obj_idh.shape[0] == 1:
                obj_idh = obj_idh.view(-1, 1).repeat(points.shape[0], 1)   #crée un tenseur colonne avec autant d'élément que de lignes dans points
            else:
                obj_idh = obj_idh.view(-1, 1)

            one_hot = torch.zeros(points.shape[0], 16).scatter_(1, obj_idh.cpu().long(), 1)  #crée un tenseur 2D avec autant de lignes que points, et 16 colonnes. remplit le tenseur avec les éléments des 1 selon les colonnes
      #indiquées dans obj_idh
            one_hot = one_hot.cuda() ## the pre-defined category ID



            pred_seg, box_pred_, feavecs = classifier_seg3D(points_n, one_hot)  #sortie du module de segmentation 3D pour pour les entrées ground-truth. pred_seg est la prédiction pour la segmentation 3D

            pred_choice = pred_seg.data.max(2)[1]  #maximum selon la dimension 2 de pred_seg
            # print(pred_choice[0])
            p = pred_choice  # [0].cpu().numpy() B N
            N_seg = 1000  #à modifier?
            pts_s = torch.zeros(points.shape[0], N_seg, 3)  #tenseur 3D de zéros avec 3 channels, autant de lignes que points et 1000 colonnes
            box_pred = torch.zeros(points.shape[0], N_seg, 3)  #tenseur 3D de zéros avec 3 channels, autant de lignes que points et 1000 colonnes
            pts_sv = torch.zeros(points.shape[0], N_seg, 3)  #tenseur 3D de zéros avec 3 channels, autant de lignes que points et 1000 colonnes
            feat = torch.zeros(points.shape[0], N_seg, feavecs.shape[2])  #tenseur 3D de zéros avec 3 channels, autant de lignes que points et 1000 colonnes


            corners0 = torch.zeros((points.shape[0], num_cor, 3))  #?
            if torch.cuda.is_available():
                ptsori = ptsori.cuda()

            Tt = np.zeros((points.shape[0], 3))  #tenseur 2D avec autant de lignes que dans points et 3 colonnes
            for ib in range(points.shape[0]):  #parcourt ??
                if len(p[ib, :].nonzero()) < 10:
                    continue

                pts_ = torch.index_select(ptsori[ib, :, 0:3], 0, p[ib, :].nonzero()[:, 0])  #??


                box_pred__ = torch.index_select(box_pred_[ib, :, :], 0, p[ib, :].nonzero()[:, 0])  #??
                feavec_ = torch.index_select(feavecs[ib, :, :], 0, p[ib, :].nonzero()[:, 0])  #??

                choice = np.random.choice(len(pts_), N_seg, replace=True)
                pts_s[ib, :, :] = pts_[choice, :]  #??
                box_pred[ib] = box_pred__[choice]  #??
                feat[ib, :, :] = feavec_[choice, :]  #??
                corners0[ib] = torch.Tensor(np.array([[0,0,0],[0,200,0],[200,0,0]]))  #??


            pts_s = pts_s.cuda()


            pts_s = pts_s.transpose(2, 1)
            cen_pred,obj_size = classifier_ce((pts_s - pts_s.mean(dim=2, keepdim=True)), obj_id) #sortie du module d'estimation de la translation. cen_pred : prédiction de la translation, obj_size : prédiction de la taille
      #de l'objet


            feavec = feat.transpose(1, 2)
            kp_m = classifier_Rot_green(feavec)
            centers = Variable(torch.Tensor((centers)))
            corners = Variable(torch.Tensor((corners)))



            #déplace les tenseurs box_pred, centers, S, corners, feat et corners0 sur GPU si disponible
            if torch.cuda.is_available():
                box_pred = box_pred.cuda()
                centers = centers.cuda()
                S = S.cuda()
                corners = corners.cuda()
                feat = feat.cuda()
                corners0 = corners0.cuda()

            loss_seg = Loss_seg3D(pred_seg.reshape(-1, pred_seg.size(-1)), target_seg.view(-1,).long())  #fonction de coût de segmentation 3D (cross-entropy) évaluée avec la prédiction pour la segmentation
            loss_res = Loss_func_ce(cen_pred, centers.float())  #fonction de coût mean square error pour la prédiction de la translation évaluée avec la prédiction de la translation
            loss_size = Loss_func_s(obj_size,S.float())  #fonction de coût mean square error pour la prédiction de la taille de l'objet évaluée avec la prédiction de la taille


            def loss_recon(a, b): #fonction de coût qui utilise la distance de Chamfer pour entraîner l'auto-encodeur des rotations des classes basé sur la convolution 3D de graphe. 
                if torch.cuda.is_available():
                    chamferdist = ChamferDistance()
                    dist1, dist2 = chamferdist(a, b)
                    loss = torch.mean(dist1) + torch.mean(dist2)
                else:
                    loss=torch.Tensor([100.0])
                return loss
            loss_vec = loss_recon(box_pred, pts_recon)  #fonction de coût qui utilise la distance de Chamfer pour entraîner l'auto-encodeur des rotations des classes basé sur la convolution 3D de graphe.

            kp_m2 = classifier_Rot_red(feat.transpose(1,2))  #sortie du module d'estimation de la rotation pour le vecteur rouge

            green_v = corners[:, 0:6].float().clone()  #obtient le vecteur vert qui représente la rotation
            red_v = corners[:, (0, 1, 2, 6, 7, 8)].float().clone()  #obtient le vecteur rouge qui représente la rotation (?)
            target = torch.tensor([[1]], dtype=torch.float).cuda()  #??

            loss_rot_g= Loss_func_Rot1(kp_m, green_v)  #fonction de coût mean square error pour l'estimation de la rotation entre le vecteur vert prédit et la ground-truth
            loss_rot_r = Loss_func_Rot2(kp_m2, red_v)  #fonction de coût mean square error pour l'estimation de la rotation entre le vecteur rouge prédit et la ground-truth


            symme=1  #terme de pondération pour la partie de la fonction de coût contenant les symétries, 1 si pas de symétrie
            if cat in ['bottle','bowl','can']:  #s'il y a une symétrie
                symme=0.0  #modifie le terme de symétrie


            Loss = loss_seg*20.0+loss_res/20.0+loss_vec/200.0+loss_size/20.0+symme*loss_rot_r/100.0+loss_rot_g/100.0  #fonction de coût totale évaluée qui combine toutes les fonctions de coût
            Loss.backward()  #calcule les gradients à partir de la fonction de coût évaluée
            optimizer.step()  #met à jour les paramètres du modèle grâce à l'optimizer

            print(cat)
            print('[%d: %d] train loss_seg: %f, loss_res: %f, loss_recon: %f, loss_size: %f, loss_rot_g: %f, '
                  'loss_rot_r: %f' % (
            epoch, i, loss_seg.item(), loss_res.item(), loss_vec.item(), loss_size.item(), loss_rot_g.item(),
            loss_rot_r.item()))


            print()

            torch.save(classifier_seg3D.state_dict(), '%s/Seg3D_last_obj%s.pth' % (opt.outf, cat))  #enregistre les poids contenus dans le module auto-encodeur de convolution de graphe 3D dans un dict
            torch.save(classifier_ce.state_dict(), '%s/Tres_last_obj%s.pth' % (opt.outf, cat))  #enregistre les poids contenus dans le module d'estimation de la translation dans un dict
            torch.save(classifier_Rot_green.state_dict(), '%s/Rot_g_last_obj%s.pth' % (opt.outf, cat))  #enregistre les poids contenus dans le d'estimation d'un des vecteurs de la rotation dans un dict
            torch.save(classifier_Rot_red.state_dict(), '%s/Rot_r_last_obj%s.pth' % (opt.outf, cat))  #enregistre les poids contenus dans le d'estimation d'un des vecteurs de la rotation dans un dict
            
            if epoch>0 and epoch %(epochs//5)== 0: ##save mid checkpoints

                torch.save(classifier_seg3D.state_dict(), '%s/Seg3D_epoch%d_obj%s.pth' % (opt.outf, epoch, cat))
                torch.save(classifier_ce.state_dict(), '%s/Tres_epoch%d_obj%s.pth' % (opt.outf, epoch, cat))
                torch.save(classifier_Rot_green.state_dict(), '%s/Rot_g_epoch%d_obj%s.pth' % (opt.outf, epoch, cat))
                torch.save(classifier_Rot_red.state_dict(), '%s/Rot_r_epoch%d_obj%s.pth' % (opt.outf, epoch, cat))




