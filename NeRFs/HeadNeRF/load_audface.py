import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2


def load_audface_data(basedir, testskip=1, test_file=None, aud_file=None):
    if test_file is not None:
        with open(os.path.join(basedir, test_file)) as fp:
            meta = json.load(fp)
        poses = []
        auds = []
        aud_features = np.load(os.path.join(basedir, aud_file))
        for frame in meta['frames'][::testskip]:
            poses.append(np.array(frame['transform_matrix']))
            auds.append(aud_features[frame['frame_id']])
        poses = np.array(poses).astype(np.float32)
        auds = np.array(auds).astype(np.float32)
        bc_img = imageio.imread(os.path.join(basedir, 'bc.jpg'))
        H, W = bc_img.shape[0], bc_img.shape[1]
        focal, cx, cy = float(meta['focal_length']), float(
            meta['cx']), float(meta['cy'])
        return poses, auds, bc_img, [H, W, focal, cx, cy]

    splits = ['train', 'val']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)
    all_imgs = []
    all_poses = []
    all_auds = []
    all_sample_rects = []
    aud_features = np.load(os.path.join(basedir, 'aud.npy'))
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        auds = []
        sample_rects = []
        mouth_rects = []
        #exps = []
        if s == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip

        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, 'head_imgs',
                                 str(frame['img_id']) + '.jpg')
            imgs.append(fname)
            poses.append(np.array(frame['transform_matrix']))
            auds.append(
                aud_features[min(frame['aud_id'], aud_features.shape[0]-1)])
            sample_rects.append(np.array(frame['face_rect'], dtype=np.int32))
        imgs = np.array(imgs)
        poses = np.array(poses).astype(np.float32)
        auds = np.array(auds).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
        all_auds.append(auds)
        all_sample_rects.append(sample_rects)

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(len(splits))]
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    auds = np.concatenate(all_auds, 0)
    sample_rects = np.concatenate(all_sample_rects, 0)

    bc_img = imageio.imread(os.path.join(basedir, 'bc.jpg'))

    H, W = bc_img.shape[:2]
    focal, cx, cy = float(meta['focal_len']), float(
        meta['cx']), float(meta['cy'])

    return imgs, poses, auds, bc_img, [H, W, focal, cx, cy], sample_rects, sample_rects, i_split
