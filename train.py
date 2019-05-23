#! /home/students/huanyu/miniconda2/envs/voxel/bin/python3.6
"""
train atlas-based alignment with CVPR2018 version of VoxelMorph
"""

# python imports
import os
import glob
import sys
import random
from argparse import ArgumentParser

# third-party imports
import tensorflow as tf
import numpy as np
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import multi_gpu_model
from PIL import Image
import matplotlib.pyplot as plt

# project imports
import datagenerators
import networks
import losses

sys.path.append('../ext/neuron')
import neuron.callbacks as nrn_gen


def train(data_dir,
          val_data_dir,
          atlas_file,
          val_atlas_file,
          model,
          model_dir,
          gpu_id,
          lr,
          nb_epochs,
          reg_param,
          gama_param,
          steps_per_epoch,
          batch_size,
          load_model_file,
          data_loss,
          seg_dir=None,  # one file
          val_seg_dir=None,
          Sf_file=None,  # one file
          val_Sf_file=None,
          auxi_label=None,
          initial_epoch=0):
    """
    model training function
    :param data_dir: folder with npz files for each subject.
    :param atlas_file: atlas filename. So far we support npz file with a 'vol' variable
    :param model: either vm1 or vm2 (based on CVPR 2018 paper)
    :param model_dir: the model directory to save to
    :param gpu_id: integer specifying the gpu to use
    :param lr: learning rate
    :param n_iterations: number of training iterations
    :param reg_param: the smoothness/reconstruction tradeoff parameter (lambda in CVPR paper)
    :param steps_per_epoch: frequency with which to save models
    :param batch_size: Optional, default of 1. can be larger, depends on GPU memory and volume size
    :param load_model_file: optional h5 model file to initialize with
    :param data_loss: 'mse' or 'ncc
    :param auxi_label: whether to use auxiliary informmation during the training
    """

    # load atlas from provided files. The atlas we used is 160x192x224.
    # atlas_file = 'D:/voxel/data/t064.tif'
    atlas = Image.open(atlas_file)   # is a TiffImageFile _size is (628, 690)
    atlas_vol = np.array(atlas)[np.newaxis, ..., np.newaxis]  # is a ndarray, shape is (1, 690, 628, 1)
                                                          # new = Image.fromarray(X) new.size is (628, 690)
    vol_size = atlas_vol.shape[1:-1]  # (690, 628)
    print(vol_size)

    val_atlas = Image.open(val_atlas_file)  # is a TiffImageFile _size is (628, 690)
    val_atlas_vol = np.array(val_atlas)[np.newaxis, ..., np.newaxis]  # is a ndarray, shape is (1, 690, 628, 1)
    # new = Image.fromarray(X) new.size is (628, 690)
    val_vol_size = val_atlas_vol.shape[1:-1]  # (690, 628)
    print(val_vol_size)

    Sm = Image.open(seg_dir)  # is a TiffImageFile _size is (628, 690)
    Sm_ = np.array(Sm)[np.newaxis, ..., np.newaxis]

    val_Sm = Image.open(val_seg_dir)  # is a TiffImageFile _size is (628, 690)
    val_Sm_ = np.array(val_Sm)[np.newaxis, ..., np.newaxis]

    # prepare data files
    # for the CVPR and MICCAI papers, we have data arranged in train/validate/test folders
    # inside each folder is a /vols/ and a /asegs/ folder with the volumes
    # and segmentations. All of our papers use npz formated data.
    # data_dir = D:/voxel/data/01
    train_vol_names = data_dir # glob.glob(os.path.join(data_dir, '*.tif'))   # is a list contain file path(name)
    # random.shuffle(train_vol_names)  # shuffle volume list    tif
    assert len(train_vol_names) > 0, "Could not find any training data"

    val_vol_names = val_data_dir  # glob.glob(os.path.join(data_dir, '*.tif'))   # is a list contain file path(name)
    # random.shuffle(train_vol_names)  # shuffle volume list    tif
    assert len(val_vol_names) > 0, "Could not find any training data"

    # UNET filters for voxelmorph-1 and voxelmorph-2,
    # these are architectures presented in CVPR 2018
    nf_enc = [16, 32, 32, 32]
    if model == 'vm1':
        nf_dec = [32, 32, 32, 32, 8, 8]
    elif model == 'vm2':
        nf_dec = [32, 32, 32, 32, 32, 16, 16]
    else:  # 'vm2double':
        nf_enc = [f * 2 for f in nf_enc]
        nf_dec = [f * 2 for f in [32, 32, 32, 32, 32, 16, 16]]

    assert data_loss in ['mse', 'cc', 'ncc'], 'Loss should be one of mse or cc, found %s' % data_loss
    if data_loss in ['ncc', 'cc']:
        data_loss = losses.NCC().loss

    if Sf_file is not None:
        Sf = Image.open(Sf_file)
        Sf_=np.array(Sf)[np.newaxis, ..., np.newaxis]

    if val_Sf_file is not None:
        val_Sf = Image.open(val_Sf_file)
        val_Sf_ = np.array(val_Sf)[np.newaxis, ..., np.newaxis]


        # prepare model folder
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # GPU handling
    gpu = '/gpu:%d' % 0  # gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))
    #gpu = gpu_id

    # data generator
    nb_gpus = len(gpu_id.split(','))    # 1
    assert np.mod(batch_size, nb_gpus) == 0, \
        'batch_size should be a multiple of the nr. of gpus. ' + \
        'Got batch_size %d, %d gpus' % (batch_size, nb_gpus)

    train_example_gen = datagenerators.example_gen(train_vol_names, batch_size=batch_size)   # it is a list contain a ndarray
    atlas_vol_bs = np.repeat(atlas_vol, batch_size, axis=0)  # is a ndarray, if batch_size is 2, shape is (2, 690, 628, 1)
    cvpr2018_gen = datagenerators.cvpr2018_gen(train_example_gen, atlas_vol_bs, batch_size=batch_size)

    val_example_gen = datagenerators.example_gen(val_vol_names,batch_size=batch_size)  # it is a list contain a ndarray
    val_atlas_vol_bs = np.repeat(val_atlas_vol, batch_size, axis=0)  # is a ndarray, if batch_size is 2, shape is (2, 690, 628, 1)
    val_cvpr2018_gen = datagenerators.cvpr2018_gen(val_example_gen, val_atlas_vol_bs, batch_size=batch_size)

    # prepare the model
    with tf.device(gpu):
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        # prepare the model
        # in the CVPR layout, the model takes in [image_1, image_2] and outputs [warped_image_1, flow]
        # in the experiments, we use image_2 as atlas

        model = networks.cvpr2018_net(vol_size, nf_enc, nf_dec)

        # load initial weights
        if load_model_file is not None:
            print('loading', load_model_file)
            model.load_weights(load_model_file)

        # save first iteration
        model.save(os.path.join(model_dir, '%02d.h5' % initial_epoch))

        # if auxi_label is not None:
        #     print('yes')
        #     loss_model= [data_loss, losses.Grad('l2').loss, losses.Lseg()._lseg(Sf_) ]    ##########################
        #     loss_weight= [1.0, reg_param, gama_param]
        # else:
        loss_model = [data_loss, losses.Grad(gama_param, Sf_, Sm_, penalty='l2').loss]  # real gama: reg_param*gama_param
        loss_weight = [1.0, reg_param]

        # reg_param_tensor = tf.constant(5, dtype=tf.float32)
        # metrics_2 = losses.Grad(gama_param, val_Sf_, val_Sm_, penalty='l2', flag_vali = True).loss   # reg_param



    # prepare callbacks
    save_file_name = os.path.join(model_dir, '{epoch:02d}.h5')

    # fit generator
    with tf.device(gpu):

        # multi-gpu support
        if nb_gpus > 1:
            save_callback = nrn_gen.ModelCheckpointParallel(save_file_name)
            mg_model = multi_gpu_model(model, gpus=nb_gpus)

        # single-gpu
        else:
            save_callback = ModelCheckpoint(save_file_name)
            mg_model = model

        # compile
        mg_model.compile(optimizer=Adam(lr=lr),
                         loss=loss_model,
                         loss_weights=loss_weight)
                         # metrics={'y': data_loss, 'flow': metrics_2})

        # fit
        history = mg_model.fit_generator(cvpr2018_gen,
                               initial_epoch=initial_epoch,
                               epochs=nb_epochs,
                               callbacks=[save_callback],
                               steps_per_epoch=steps_per_epoch,
                               validation_data= val_cvpr2018_gen,
                               validation_steps=1,
                               verbose=2)

        # plot
        print(history.history.keys())
        plt.plot(history.history['loss'])
        # plt.plot(history.history['metrics'])
        plt.title('cvpr_auxi_loss')
        plt.ylabel('loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'])
        plt.show()



if __name__ == "__main__":
    data_dir_ori = '/home/students/huanyu/PycharmProjects/voxel/data/02_Converted_small'
    list_names = glob.glob(os.path.join(data_dir_ori, '*.tif'))
    list_num = len(list_names)
    print('number of moving image is:', list_num)
    # choose_atlas = np.random.randint(5, list_num-5)   # [ )

    seg_dir_ori= '/home/students/huanyu/PycharmProjects/voxel/data/tra_02'
    seg_list_names = glob.glob(os.path.join(seg_dir_ori, '*.tif'))

    model = 'vm2'
    gpu_id = '0'
    lr = 1e-4
    nb_epochs = 50
    reg_param = 0.01
    gama_param = 1
    steps_per_epoch = 50
    batch_size = 1  # it seems only can be 1

    data_loss = 'mse'
    auxi_label = 1

    for m in range(0, 3):   # 0, 100
        choose_atlas = m
        data_dir = list_names[choose_atlas + 1]
        # data_dir = 'D:/voxel/data/01_Converted'
        seg_dir = seg_list_names[choose_atlas + 1]
        # del seg_dir[5]
        atlas_file = list_names[choose_atlas] # 'D:/voxel/data/40.tif'
        Sf_file = seg_list_names[choose_atlas]

        if m%4 == 0:
            val_data_dir = list_names[choose_atlas + 102]
            val_seg_dir = seg_list_names[choose_atlas + 102]

            val_atlas_file = list_names[choose_atlas + 101]
            val_Sf_file = seg_list_names[choose_atlas + 101]

        model_dir = '/home/students/huanyu/PycharmProjects/voxel/models_auxi'
        model_dir = os.path.join(model_dir, str(m))
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

        load_model_file = '/home/students/huanyu/PycharmProjects/voxel/models_auxi'  # 'D:/voxel/models/1/50.h5'
        load_model_file = os.path.join(load_model_file, str(m - 1))
        load_model_file = os.path.join(load_model_file, '50.h5')

        if not os.path.isfile(load_model_file):
            load_model_file = None

        train(data_dir,
              val_data_dir,
              atlas_file,
              val_atlas_file,
              model,
              model_dir,
              gpu_id,
              lr,
              nb_epochs,
              reg_param,
              gama_param,
              steps_per_epoch,
              batch_size,
              load_model_file,
              data_loss,
              seg_dir=seg_dir,    # one file
              val_seg_dir = val_seg_dir,
              Sf_file=Sf_file,     # one file
              val_Sf_file = val_Sf_file,
              auxi_label=auxi_label

              )

    for n in range(100, list_num + 50):   # 100, 200
        choose_atlas = n - 99
        data_dir = list_names[choose_atlas - 1]
        # data_dir = 'D:/voxel/data/01_Converted'
        seg_dir = seg_list_names[choose_atlas - 1]
        # del seg_dir[5]

        atlas_file = list_names[choose_atlas]  # 'D:/voxel/data/40.tif'
        Sf_file = seg_list_names[choose_atlas]

        if n%4 == 0:
            val_data_dir = list_names[choose_atlas + 100]
            val_seg_dir = seg_list_names[choose_atlas + 100]

            val_atlas_file = list_names[choose_atlas + 101]
            val_Sf_file = seg_list_names[choose_atlas + 101]
        model_dir = '/home/students/huanyu/PycharmProjects/voxel/models_auxi'
        model_dir = os.path.join(model_dir, str(n))
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

        load_model_file = '/home/students/huanyu/PycharmProjects/voxel/models_auxi'  # 'D:/voxel/models/1/50.h5'
        load_model_file = os.path.join(load_model_file, str(n - 1))
        load_model_file = os.path.join(load_model_file, '50.h5')

        if not os.path.isfile(load_model_file):
            load_model_file = None

        train(data_dir,
              val_data_dir,
              atlas_file,
              val_atlas_file,
              model,
              model_dir,
              gpu_id,
              lr,
              nb_epochs,
              reg_param,
              gama_param,
              steps_per_epoch,
              batch_size,
              load_model_file,
              data_loss,
              seg_dir=seg_dir,  # one file
              val_seg_dir=val_seg_dir,
              Sf_file=Sf_file,  # one file
              val_Sf_file=val_Sf_file,
              auxi_label=auxi_label

              )

