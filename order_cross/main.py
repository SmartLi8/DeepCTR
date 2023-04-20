
import os
import numpy as np
import pandas as pd
import gc
from datetime import datetime, date, timedelta
import tensorflow as tf
# from tensorflow.python.keras.optimizers import Adam
from deepctr.models.multitask import MMOE
from feat_preprocess import feat_config, build_input_fn_from_csv, build_model_input_feat
from constant import *
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)
# from pyspark.sql import SparkSession
#
# spark = SparkSession.builder.appName("order_cross") \
#     .config("hive.metastore.local", "false") \
#     .config("spark.io.compression.codec", "snappy") \
#     .config("spark.sql.execution.arrow.enabled", "true") \
#     .enableHiveSupport() \
#     .getOrCreate()

download = False
model_feats, sparse_feats, dense_feats, inputdim_dict, bin_dict, share_emb_dict = feat_config(HDFS_LIST, LOCAL_LIST,
                                                                                              download=download)
train_list = [os.path.join(LOCAL_PATH_TRAIN, file) for file in tf.io.gfile.listdir(LOCAL_PATH_TRAIN)]
test_list = [os.path.join(LOCAL_PATH_TEST, file) for file in tf.io.gfile.listdir(LOCAL_PATH_TEST)]
predict_list = [os.path.join(LOCAL_PATH_PREDICT, file) for file in tf.io.gfile.listdir(LOCAL_PATH_PREDICT)]
select_columns = model_feats + LABEL_NAMES
print(len(CSV_COLUMNS))
ds_train = build_input_fn_from_csv(train_list, CSV_COLUMNS, select_columns)
ds_test = build_input_fn_from_csv(test_list, CSV_COLUMNS, select_columns, shuffle=False)
# ds_pred = build_input_fn_from_csv(predict_list, CSV_COLUMNS, ['id'] + select_columns, shuffle=False,
#                                   sloppy=False)
ds_train = ds_train.prefetch(5)
ds_test = ds_test.prefetch(5)
# ds_pred = ds_pred.prefetch(5)
feature_columns, feature_names = build_model_input_feat(sparse_feats, dense_feats, inputdim_dict, share_emb_dict,
                                                        bin_dict)

model_params = {
    'dnn_feature_columns': feature_columns,
    'num_experts': 7,
    'task_types': ('binary', 'binary', 'binary', 'binary', 'binary', 'binary', 'binary'),
    'task_names': TASK_NAMES,
    'dnn_dropout': 0.2,
    'expert_dnn_hidden_units': (512, 256, 256, 128),
    'tower_dnn_hidden_units': (128, 64)
}

model = MMOE(**model_params)
callback1 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=1)
# opt = Adam(learning_rate=0.0003)
model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['AUC'])
# os.system("hadoop fs -get {0} {1} ".format('/ns-dcbi/dm/tmp/online_train_model/order_cross_mmoe', '.'))
# model = tf.keras.models.load_model('data/order_cross_mmoe')
history = model.fit(ds_train, epochs=EPOCH, validation_data=ds_test, callbacks=[callback1])
print(history.history)
model.save('./model/order_cross_mmoe')
# os.system(
#     "hadoop fs -put {0} {1} ".format('./model/order_cross_mmoe', '/ns-dcbi/dm/tmp/online_train_model/order_cross_mmoe'))
# prob = model.predict(ds_pred)
# id_dataset = ds_pred.map(lambda x, y: x['id'])
# idset = list(id_dataset.as_numpy_iterator())
# res_arr = np.concatenate((np.concatenate(idset).reshape(-1, 1), np.concatenate(prob, axis=1)), axis=1)
# df_result = pd.DataFrame(res_arr, columns=['id'] + TASK_NAMES)
# spark.createDataFrame(df_result).write.mode('overwrite').format('hive').saveAsTable(
#     'tmp_dm.tmp_lty_order_cross_predict_data_result')
