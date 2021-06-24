import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2


def load_audface_data(basedir, testskip=1, test_file=None, aud_file=None, test_size=-1):
    if test_file is not None:
        with open(os.path.join(basedir, test_file)) as fp:
            meta = json.load(fp)
        poses = []
        auds = []
        aud_features = np.load(os.path.join(basedir, aud_file))
        cur_id = 0
        for frame in meta['frames'][::testskip]:
            poses.append(np.array(frame['transform_matrix']))
            aud_id = cur_id
            auds.append(aud_features[aud_id])
            cur_id = cur_id + 1
            if cur_id == aud_features.shape[0] or cur_id == test_size:
                break
        poses = np.array(poses).astype(np.float32)
        auds = np.array(auds).astype(np.float32)
        bc_img = imageio.imread(os.path.join(basedir, 'bc.jpg'))
        H, W = bc_img.shape[0], bc_img.shape[1]
        focal, cx, cy = float(meta['focal_len']), float(
            meta['cx']), float(meta['cy'])
        return poses, auds, bc_img, [H, W, focal, cx, cy]

    splits = ['train', 'val']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)
    all_com_imgs = []
    all_poses = []
    all_auds = []
    all_sample_rects = []
    aud_features = np.load(os.path.join(basedir, 'aud.npy'))
    counts = [0]
    for s in splits:
        meta = metas[s]
        com_imgs = []
        poses = []
        auds = []
        sample_rects = []
        if s == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip

        for frame in meta['frames'][::skip]:
            filename = os.path.join(basedir, 'com_imgs',
                                    str(frame['img_id']) + '.jpg')
            com_imgs.append(filename)
            poses.append(np.array(frame['transform_matrix']))
            auds.append(
                aud_features[min(frame['aud_id'], aud_features.shape[0]-1)])
            sample_rects.append(np.array(frame['face_rect'], dtype=np.int32))
        com_imgs = np.array(com_imgs)
        poses = np.array(poses).astype(np.float32)
        auds = np.array(auds).astype(np.float32)
        counts.append(counts[-1] + com_imgs.shape[0])
        all_com_imgs.append(com_imgs)
        all_poses.append(poses)
        all_auds.append(auds)
        all_sample_rects.append(sample_rects)
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(len(splits))]
    com_imgs = np.concatenate(all_com_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    auds = np.concatenate(all_auds, 0)
    sample_rects = np.concatenate(all_sample_rects, 0)

    bc_img = imageio.imread(os.path.join(basedir, 'bc.jpg'))

    H, W = bc_img.shape[:2]
    focal, cx, cy = float(meta['focal_len']), float(
        meta['cx']), float(meta['cy'])

    return com_imgs, poses, auds, bc_img, [H, W, focal, cx, cy], \
        sample_rects, i_split


def load_test_data(basedir, aud_file, test_pose_file='transforms_train.json',
                   testskip=1, test_size=-1, aud_start=0):
    with open(os.path.join(basedir, test_pose_file)) as fp:
        meta = json.load(fp)
    poses = []
    auds = []
    aud_features = np.load(aud_file)
    aud_ids = []
    cur_id = 0
    for frame in meta['frames'][::testskip]:
        poses.append(np.array(frame['transform_matrix']))
        auds.append(
            aud_features[min(aud_start+cur_id, aud_features.shape[0]-1)])
        aud_ids.append(aud_start+cur_id)
        cur_id = cur_id + 1
        if cur_id == test_size or cur_id == aud_features.shape[0]:
            break
    poses = np.array(poses).astype(np.float32)
    auds = np.array(auds).astype(np.float32)
    bc_img = imageio.imread(os.path.join(basedir, 'bc.jpg'))
    H, W = bc_img.shape[0], bc_img.shape[1]
    focal, cx, cy = float(meta['focal_len']), float(
        meta['cx']), float(meta['cy'])

    with open(os.path.join(basedir, 'transforms_train.json')) as fp:
        meta_torso = json.load(fp)
    torso_pose = np.array(meta_torso['frames'][0]['transform_matrix'])
    return poses, auds, bc_img, [H, W, focal, cx, cy], aud_ids, torso_pose
