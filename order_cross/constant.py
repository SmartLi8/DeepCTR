# constant
FEAT_CONFIG_FILE = 'feat_config_order_cross.csv'
HDFS_PATH_TRAIN = '/ns-dcbi/tmp/tmp_lty_order_cross_train_data'
HDFS_PATH_TEST = '/ns-dcbi/tmp/tmp_lty_order_cross_test_data'
HDFS_PATH_PREDICT = '/ns-dcbi/tmp/tmp_lty_order_cross_predict_data'
HDFS_PATH_BUCKETBIN = '/ns-dcbi/tmp/tmp_lty_order_cross_dense_feature_bucket_bin'
LOCAL_PATH_TRAIN = './data/tmp_lty_order_cross_train_data'
LOCAL_PATH_TEST = './data/tmp_lty_order_cross_test_data'
LOCAL_PATH_PREDICT = './data/tmp_lty_order_cross_predict_data'
LOCAL_PATH_BUCKETBIN = './data/tmp_lty_order_cross_dense_feature_bucket_bin'
RESULT_TABLE = 'tmp_dm.tmp_lty_order_cross_predict_data_result'

LABEL_NAMES = ['label_1', 'label_2', 'label_3', 'label_4', 'label_5', 'label_6', 'label_7']
TASK_NAMES = ['hotel', 'scenery', 'fly', 'intelfly', 'zhuanche', 'sfcar', 'tour']
CSV_COLUMNS = ['unionid', 'sampletime', 'sampledate', 'label_1', 'label_2', 'label_3', 'label_4', 'label_5',
               'label_6', 'label_7', 'productid', 'start_cityid', 'end_cityid', 'useday', 'sex', 'age',
               'agerange', 'resident_cityid', 'resident_citylevel', 'member_level', 'is_local', 'is_student',
               'is_parent', 'is_businessman', 'is_worker', 'workdayindicator', 'hour', 'time_quantum', 'week',
               'weather', 'most_poi_name', 'most_poi_type3', 'geohash6', 'poi_name', 'poi_type3',
               'user_sfcar_viewcnt_14d', 'user_fightingtickets_viewcnt_14d', 'user_train_shorttrip_cnt_90d',
               'user_all_ordercnt_180d', 'user_all_ordercnt_7d', 'user_hotel_ordercnt_730d', 'user_hotel_ordercnt_7d',
               'user_fly_ordercnt_730d', 'user_fly_ordercnt_7d', 'user_scenery_ordercnt_730d',
               'user_scenery_ordercnt_30d', 'user_scenery_ordercnt_7d', 'user_train_ordercnt_180d',
               'user_train_ordercnt_30d', 'user_train_ordercnt_7d', 'user_bus_ordercnt_730d', 'user_bus_ordercnt_30d',
               'user_bus_ordercnt_7d', 'user_car_ordercnt_730d', 'user_car_ordercnt_30d', 'user_car_ordercnt_7d',
               'user_hotel_medianprice_730d', 'user_fly_medianprice_730d', 'user_scenery_medianprice_730d',
               'user_train_medianprice_730d', 'user_car_medianprice_730d', 'user_hotel_lastorderdays',
               'user_fly_lastorderdays', 'user_scenery_lastorderdays', 'user_train_lastorderdays',
               'user_car_lastorderdays', 'is_hotel_trip', 'is_fly_trip', 'is_scenery_trip', 'is_train_trip',
               'is_bus_trip', 'is_car_trip', 'user_all_viewday_3d', 'user_hotel_viewday_3d', 'user_fly_viewday_3d',
               'user_scenery_viewday_3d', 'user_train_viewday_3d', 'user_bus_viewday_3d', 'user_car_viewday_3d',
               'user_hotel_listpagecnt_3d', 'user_hotel_detailpagecnt_3d', 'user_hotel_orderinfopagecnt_3d',
               'user_fly_listpagecnt_3d', 'user_fly_detailpagecnt_3d', 'user_fly_orderinfopagecnt_3d',
               'user_scenery_listpagecnt_3d', 'user_scenery_detailpagecnt_3d', 'user_scenery_orderinfopagecnt_3d',
               'user_train_listpagecnt_3d', 'user_train_detailpagecnt_3d', 'user_train_orderinfopagecnt_3d',
               'user_bus_listpagecnt_3d', 'user_bus_detailpagecnt_3d', 'user_bus_orderinfopagecnt_3d',
               'user_car_listpagecnt_3d', 'user_car_detailpagecnt_3d', 'user_car_orderinfopagecnt_3d',
               'user_all_viewday_14d', 'user_hotel_viewday_14d', 'user_fly_viewday_14d', 'user_scenery_viewday_14d',
               'user_train_viewday_14d', 'user_bus_viewday_14d', 'user_car_viewday_14d', 'user_hotel_listpagecnt_14d',
               'user_hotel_detailpagecnt_14d', 'user_hotel_orderinfopagecnt_14d', 'user_fly_listpagecnt_14d',
               'user_fly_detailpagecnt_14d', 'user_fly_orderinfopagecnt_14d', 'user_scenery_listpagecnt_14d',
               'user_scenery_detailpagecnt_14d', 'user_scenery_orderinfopagecnt_14d', 'user_train_listpagecnt_14d',
               'user_train_detailpagecnt_14d', 'user_train_orderinfopagecnt_14d', 'user_bus_listpagecnt_14d',
               'user_bus_detailpagecnt_14d', 'user_bus_orderinfopagecnt_14d', 'user_car_listpagecnt_14d',
               'user_car_detailpagecnt_14d', 'user_car_orderinfopagecnt_14d',
               'user_intelfly_detailpagecnt_today', 'user_intelfly_ordercnt_today', 'user_zhuanche_ordercnt_today',
               'user_sfcar_ordercnt_today', 'user_tour_ordercnt_today']
EMB_DIM = 'auto'
BATCH_SIZE = 1024
EPOCH = 2
HDFS_LIST = [HDFS_PATH_TRAIN, HDFS_PATH_TEST, HDFS_PATH_PREDICT, HDFS_PATH_BUCKETBIN]
LOCAL_LIST = [LOCAL_PATH_TRAIN, LOCAL_PATH_TEST, LOCAL_PATH_PREDICT, LOCAL_PATH_BUCKETBIN]
