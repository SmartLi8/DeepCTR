import sys
sys.path.append("..")

import pandas as pd
import os
import tensorflow as tf
from constant import FEAT_CONFIG_FILE, LOCAL_PATH_BUCKETBIN, BATCH_SIZE, EMB_DIM
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names, VarLenSparseFeat


def feat_config(hdfs_list=[], local_list=None, download=False):
    # 下载数据集
    if local_list is None:
        local_list = []
    if download:
        for hdfs, local in zip(hdfs_list, local_list):
            os.system(" rm -rf {0} ".format(local))
            os.system("hadoop fs -get {0} {1} ".format(hdfs, local))
    # 读取特征配置
    df_feat = pd.read_csv(FEAT_CONFIG_FILE)
    df_feat.dropna(subset=['feature'], inplace=True)
    model_feats = df_feat['feature'].tolist()
    sparse_feats = df_feat.loc[df_feat.features_type == 'SparseFeat', 'feature'].tolist()
    dense_feats = df_feat.loc[df_feat.features_type == 'DenseFeat', 'feature'].tolist()
    df_feat['vocabulary_size'] = df_feat['vocabulary_size'].fillna(1)
    df_feat['vocabulary_size'] = df_feat['vocabulary_size'].astype('int32')
    # embedding维度大小
    inputdim_dict = dict(zip(df_feat.loc[df_feat.features_type == 'SparseFeat']['feature'],
                             df_feat.loc[df_feat.features_type == 'SparseFeat']['vocabulary_size']))
    # 连续特征分桶区间
    df_bin = pd.read_csv(os.path.join(LOCAL_PATH_BUCKETBIN, os.listdir(LOCAL_PATH_BUCKETBIN)[0]), sep='\001',
                         names=dense_feats, na_values=r'\N')
    bin_dict = dict()
    for c in dense_feats:
        bin_dict[c] = sorted([0.0] + list(set([float(i) for i in df_bin[c][0].split('|')])))
        inputdim_dict[c] = len(bin_dict[c]) + 1
    df_emb = df_feat.dropna(subset=['emb_name'])
    share_emb_dict = dict(zip(df_emb['feature'], df_emb['emb_name']))
    return model_feats, sparse_feats, dense_feats, inputdim_dict, bin_dict, share_emb_dict


# tensorflow.data读取csv文件
def build_input_fn_from_csv(csv_file, column_names, select_columns, shuffle=True, sloppy=True):
    ds = tf.data.experimental.make_csv_dataset(
        csv_file, header=False, na_value=r'\N', column_names=column_names,
        batch_size=BATCH_SIZE, field_delim='\001', select_columns=select_columns,
        sloppy=sloppy, shuffle=shuffle, num_parallel_reads=4, num_epochs=1, ignore_errors=True
    )
    return ds.map(map_label)


# 特征和标签拆分
def map_label(x):
    label = (x['label_1'], x['label_2'], x['label_3'], x['label_4'], x['label_5'], x['label_6'], x['label_7'])
    x.pop('label_1')
    x.pop('label_2')
    x.pop('label_3')
    x.pop('label_4')
    x.pop('label_5')
    x.pop('label_6')
    x.pop('label_7')
    return x, label


def build_model_input_feat(sparse_feats, dense_feats, inputdim_dict, share_emb_dict, bin_dict):
    sparse_feat_cols = []
    for f in sparse_feats:
        if f in share_emb_dict:
            sf = SparseFeat(f, inputdim_dict[f], dtype='int32', embedding_dim=EMB_DIM, embedding_name=share_emb_dict[f])
        else:
            sf = SparseFeat(f, inputdim_dict[f], dtype='int32', embedding_dim=EMB_DIM)
        sparse_feat_cols.append(sf)
    dense_feat_cols = [
        SparseFeat(f, inputdim_dict[f], embedding_dim=EMB_DIM, dtype='float32', discretization=bin_dict[f]) for f in
        dense_feats]

    feature_columns = sparse_feat_cols + dense_feat_cols
    feature_names = get_feature_names(feature_columns)
    return feature_columns, feature_names
