# import csv
# import numpy as np
# import pickle
#
# def parse_pkl_to_dict(pkl_path):
#     with open(pkl_path, 'rb') as file:
#         data = pickle.load(file)
#     return data
#
# def main(input_pkl, output_csv):
#     data = parse_pkl_to_dict(input_pkl)
#
#     # 过滤掉 Ego 数据
#     if 'Ego' in data['ids']:
#         ego_index = data['ids'].index('Ego')
#         filtered_ids = [id for id in data['ids'] if id != 'Ego']
#         filtered_positions = np.delete(data['positions'], ego_index, axis=0)
#         filtered_headings = np.delete(data['headings'], ego_index, axis=0)
#         filtered_valid_mask = np.delete(data['valid_mask'], ego_index, axis=0)
#         filtered_types = np.delete(data['types'], ego_index)
#     else:
#         filtered_ids = data['ids']
#         filtered_positions = data['positions']
#         filtered_headings = data['headings']
#         filtered_valid_mask = data['valid_mask']
#         filtered_types = data['types']
#
#     with open(output_csv, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['frame', 'id', 'x', 'y', 'heading', 'type'])
#
#         for idx, obj_id in enumerate(filtered_ids):
#             for frame in range(filtered_valid_mask.shape[1]):
#                 if filtered_valid_mask[idx, frame]:
#                     x, y = filtered_positions[idx, frame]
#                     heading = filtered_headings[idx, frame]
#                     obj_type = filtered_types[idx]
#                     writer.writerow([frame + 1, obj_id, x, y, heading, obj_type])
#
# if __name__ == '__main__':
#     main(r"D:\Github\Onsite_rule_driven_model-main\sample\scene\mixed_952_32_1\mixed_952_32_1_gt.pkl", r'D:\Desktop\output.csv')

import tensorflow as tf

# 定义 TFRecord 文件路径
file_pattern = r"D:\Desktop\motion\motion\uncompressed_tf_example_testing_testing_tfexample.tfrecord-00000-of-00150"

# 创建 TFRecord 数据集
dataset = tf.data.TFRecordDataset(tf.data.Dataset.list_files(file_pattern))

# 定义解析函数
def parse_example(example_proto):
    # 定义特征描述
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    # 解析示例
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    # 解码图像
    image = tf.io.decode_image(parsed_features['image'])
    label = parsed_features['label']
    return image, label

# 解析数据集
dataset = dataset.map(parse_example)

# 遍历数据集
for image, label in dataset.take(5):
    print(f'Image shape: {image.shape}, Label: {label.numpy()}')
