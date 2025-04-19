import csv
import xml.etree.ElementTree as ET
import numpy as np
import torch
from dask.array import shape
from pyexpat import features
import matplotlib.pyplot as plt

from scipy.spatial import KDTree

# 定义车道类型映射
lane_type_mapping = {
    'none': 0,
    'driving': 1,
    'shoulder': 2,
    'stop': 3,
    # 可以根据实际情况添加更多类型
}

# 定义道路标记类型映射
road_mark_type_mapping = {
    'none': 0,
    'solid': 1,
    'broken': 2,
    'doubleSolid': 3,
    # 可以根据实际情况添加更多类型
}

# 定义材料映射
material_mapping = {
    'standard': 0,
    'other': 1,
    # 可以根据实际情况添加更多类型
}

# 定义颜色映射
color_mapping = {
    'white': 0,
    'yellow': 1,
    # 可以根据实际情况添加更多颜色
}

# 新增交通信号灯状态映射
signal_state_mapping = {
    'red': 0,
    'yellow': 1,
    'green': 2,
    'off': 3,
}


def parse_opendrive_map(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        map_features = {
            'lanes': [],
            'road_lines': [],
            'stop_signs': [],
            'traffic_signals': [],  # 新增信号灯特征
            'speed_limits': [],  # 新增限速标志特征
            'boundary': None
        }

        # 提取地图边界
        header = root.find('header')
        x_min = float(header.get('west'))
        x_max = float(header.get('east'))
        y_min = float(header.get('south'))
        y_max = float(header.get('north'))
        map_features['boundary'] = np.array([x_min, x_max, y_min, y_max])

        # 遍历所有道路
        for road in root.findall('road'):
            road_geometry = []
            plan_view = road.find('planView')
            if plan_view is not None:
                for geom in plan_view.findall('geometry'):
                    try:
                        s = float(geom.get('s', 0.0))
                        x = float(geom.get('x', 0.0))
                        y = float(geom.get('y', 0.0))
                        hdg = float(geom.get('hdg', 0.0))
                        length = float(geom.get('length', 0.0))
                        # 计算终点坐标（根据hdg和length）
                        end_x = x + length * np.cos(hdg)
                        end_y = y + length * np.sin(hdg)
                        road_geometry.append({
                            's': s,
                            'x': x,
                            'y': y,
                            'end_x': end_x,  # 新增终点坐标
                            'end_y': end_y,
                            'hdg': hdg,
                            'length': length,
                            'type': geom.tag
                        })
                    except ValueError as e:
                        print(f"Invalid geometry value: {e}")
                        continue

            # 处理车道信息
            lanes = road.find('lanes')
            if lanes is None:
                continue
            for lane_section in lanes.findall('laneSection'):
                for direction in ['center', 'right', 'left']:
                    dir_section = lane_section.find(direction)
                    if dir_section is None:
                        continue
                    for lane in dir_section.findall('lane'):
                        try:
                            lane_id = lane.get('id', '')
                            lane_type = lane.get('type', 'none')
                            width = lane.find('width')
                            width_value = float(width.get('a', 0.0)) if width is not None else 0.0
                            road_marks = []
                            for mark in lane.findall('roadMark'):
                                mark_type = mark.get('type', 'none')
                                material = mark.get('material', 'standard')
                                color = mark.get('color', 'white')
                                road_marks.append({
                                    'type': mark_type,
                                    'material': material,
                                    'color': color
                                })
                            map_features['lanes'].append({
                                'id': lane_id,
                                'type': lane_type,
                                'width': width_value,
                                'road_marks': road_marks,
                                'geometry': road_geometry
                            })
                        except Exception as e:
                            print(f"Error processing lane: {e}")
                            continue

            # 提取停车标志
            objects = road.find('objects')
            if objects is not None:
                for obj in objects.findall('object'):
                    obj_type = obj.get('type')
                    if obj_type == 'roadmark' and obj.get('name') == 'stopLocation':
                        s = float(obj.get('s'))
                        t = float(obj.get('t'))
                        map_features['stop_signs'].append({
                            'position': (s, t),
                            'orientation': float(obj.get('hdg'))
                        })
            # 新增交通信号灯提取
            signals = road.findall('.//object[@type="signal"]')
            for signal in signals:
                try:
                    s = float(signal.get('s'))
                    t = float(signal.get('t'))
                    state = signal.get('state', 'off')
                    map_features['traffic_signals'].append({
                        'position': (s, t),
                        'state': state,
                        'orientation': float(signal.get('hdg', 0.0))
                    })
                except Exception as e:
                    print(f"解析信号灯时出现错误: {e}")

            # 新增限速标志提取
            speed_limits = road.findall('.//speed')
            for limit in speed_limits:
                try:
                    s = float(limit.get('sOffset', 0.0))
                    max_speed = float(limit.get('max', 0.0))
                    unit = limit.get('unit', 'mph')
                    map_features['speed_limits'].append({
                        'sOffset': s,
                        'max_speed': max_speed,
                        'unit': unit
                    })
                except Exception as e:
                    print(f"解析限速标志时出现错误: {e}")

        return map_features
    except Exception as e:
        print(f"解析地图文件时出错: {e}")
        return None

def preprocess_lane_geometry(lanes):
    """
    预处理车道几何数据为可查询的结构
    :param lanes: 车道信息列表
    :return: KDTree、车道信息列表、所有线段列表
    """
    all_segments = []
    lane_info = []

    for lane in lanes:
        geom = lane['geometry']
        if len(geom) < 2:
            if len(geom) == 1:
                start = (geom[0]['x'], geom[0]['y'])
                hdg = geom[0]['hdg']
                length = geom[0]['length']
                end = (start[0] + length * np.cos(hdg),
                       start[1] + length * np.sin(hdg))
                geom.append({'x': end[0], 'y': end[1]})
        for i in range(len(geom)-1):
            try:
                start = (geom[i]['x'], geom[i]['y'])
                end = (geom[i + 1]['x'], geom[i + 1]['y'])
                all_segments.append((start, end))
                lane_info.append({
                    'lane_id': lane['id'],
                    'type': lane['type'],
                    'width': lane['width'],
                    'road_marks': lane['road_marks']
                })
            except Exception as e:
                print(f"Error processing segment: {e}")

                continue
        # 生成线段点（包含起点和终点）
    segment_points = []
    segment_indices = []  # 记录每个点对应的线段索引
    for idx, seg in enumerate(all_segments):
        start = seg[0]
        end = seg[1]
        # 添加起点和终点
        segment_points.extend([start, end])
        segment_indices.extend([idx, idx])  # 每个点对应原线段索引

    if not all_segments:
        print("No valid segments found in lanes")
        return None, None, None

    segment_points = np.array(segment_points)

    return KDTree(segment_points), lane_info, all_segments, segment_indices




def closest_point_on_segment(segment, point):
    """
    计算点到线段的最近点
    :param segment: 线段，由起点和终点组成的元组
    :param point: 待计算的点
    :return: 点到线段的最近点
    """
    (x1, y1), (x2, y2) = segment
    px, py = point

    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        return x1, y1

    t = ((px - x1) * dx + (py - y1) * dy) / (dx ** 2 + dy ** 2)
    t = max(0, min(1, t))

    x_proj = x1 + t * dx
    y_proj = y1 + t * dy
    return x_proj, y_proj


def get_closest_lane_features(participant_pos, kdtree, lane_info, all_segments,segment_indices):
    """
    获取最近车道的特征

    :param participant_pos: 交通参与者的位置 (x, y)
    :param kdtree: 预构建的KDTree
    :param lane_info: 车道信息列表
    :param all_segments: 所有线段列表
    :return: 包含最近车道特征的字典或None
    """
    dist, idx = kdtree.query(participant_pos)
    # if dist > 20:
    #     return None

    # 通过segment_indices找到实际对应的线段
    segment_idx = segment_indices[idx]
    segment = all_segments[segment_idx]

    # 计算真实最近点
    closest_point = closest_point_on_segment(segment, participant_pos)
    distance = np.linalg.norm(np.array(participant_pos) - closest_point)

    return {
        **lane_info[segment_idx],
        'closest_point': closest_point,
        'distance': distance
    }




def encode_lane(lane):
    features = []
    # 编码车道类型
    lane_type = lane_type_mapping.get(lane['type'], 0)
    features.append(lane_type)
    # 编码车道宽度
    features.append(lane['width'])
    # 编码道路标记
    for mark in lane['road_marks']:
        mark_type = road_mark_type_mapping.get(mark['type'], 0)
        material = material_mapping.get(mark['material'], 0)
        color = color_mapping.get(mark['color'], 0)
        features.extend([mark_type, material, color])
    # 编码几何信息
    for geom in lane['geometry']:
        features.extend([geom['x'], geom['y'], geom['hdg'], geom['length']])
    return torch.FloatTensor(features)


def encode_stop_sign(stop_sign):
    position = stop_sign['position']
    orientation = stop_sign['orientation']
    features = [*position, orientation]
    return torch.FloatTensor(features)


def encode_map_features(map_features):
    encoded_lanes = [encode_lane(lane) for lane in map_features['lanes']]
    encoded_stop_signs = [encode_stop_sign(stop_sign) for stop_sign in map_features['stop_signs']]
    encoded_signals = [encode_traffic_signal(sig) for sig in map_features['traffic_signals']]  # 新增编码
    encoded_speed_limits = [encode_speed_limit(limit) for limit in map_features['speed_limits']]  # 新增编码
    boundary = torch.FloatTensor(map_features['boundary'])
    return {
        'lanes': encoded_lanes,
        'stop_signs': encoded_stop_signs,
        'boundary': boundary
    }

# 新增信号灯编码函数
def encode_traffic_signal(signal):
    position = signal['position']
    state = signal_state_mapping.get(signal['state'], 3)
    orientation = signal['orientation']
    return torch.FloatTensor([*position, state, orientation])

# 新增限速标志编码函数
def encode_speed_limit(limit):
    # 转换为统一单位（示例转换为 km/h）
    speed = limit['max_speed']
    if limit['unit'] == 'mph':
        speed *= 1.60934
    return torch.FloatTensor([limit['sOffset'], speed])





def read_participant_file(file_path):
    """
    读取交通参与者文件，获取每一帧的信息
    :param file_path: 交通参与者文件路径
    :return: 每一帧的交通参与者信息列表
    """
    frames = [[] for _ in range(32)]
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行
        for row in reader:
            frame = int(row[0])
            x = float(row[2])
            y = float(row[3])
            heading = float(row[4])
            type_id = int(row[5])
            frames[frame].append([x, y, heading, type_id])
    return frames

def encode_participant_features(participant, map_features):
    """
    合并交通参与者特征和地图特征
    :param participant: 交通参与者的特征
    :param map_features: 地图特征
    :return: 合并后的特征张量
    """
    # 交通参与者原始特征
    x, y, heading, type_id = participant

    # 地图特征
    lane_type = lane_type_mapping.get(map_features['type'], 0)
    lane_width = map_features['width']
    # 道路标记特征（示例取第一个标记）
    if map_features['road_marks']:
        mark = map_features['road_marks'][0]
        road_mark_type = road_mark_type_mapping.get(mark['type'], 0)
        road_material = material_mapping.get(mark['material'], 0)
        road_color = color_mapping.get(mark['color'], 0)
    else:
        road_mark_type = road_material = road_color = 0

    # 合并特征
    return torch.FloatTensor([
        x, y, heading, type_id,
        lane_type, lane_width,
        road_mark_type, road_material, road_color,
        map_features['distance']  # 距离最近车道的距离
    ])


def visualize_scene(all_segments, participants, closest_distances):
    """可视化场景"""
    plt.figure(figsize=(12, 12))

    # 绘制所有车道线段
    for seg in all_segments:
        x_vals = [seg[0][0], seg[1][0]]
        y_vals = [seg[0][1], seg[1][1]]
        plt.plot(x_vals, y_vals, 'b-', linewidth=1, alpha=0.5)

    # 绘制交通参与者及其最近点
    for i, (participant, distance) in enumerate(zip(participants, closest_distances)):
        x, y = participant[0], participant[1]

        # 绘制车辆位置
        plt.plot(x, y, 'ro', markersize=8)
        plt.text(x + 0.5, y + 0.5, f'V{i}', color='red')

        # 如果存在最近车道信息
        if distance is not None:
            closest_x, closest_y = distance['closest_point']
            # 绘制连接线
            plt.plot([x, closest_x], [y, closest_y], 'g--', linewidth=1)
            # 标注距离
            plt.text((x + closest_x) / 2, (y + closest_y) / 2,
                     f'd={distance["distance"]:.2f}m',
                     color='purple',
                     fontsize=8)

    plt.title("Vehicle-Lane Distance Visualization")
    plt.xlabel("X Coordinate (meters)")
    plt.ylabel("Y Coordinate (meters)")
    plt.grid(True)
    plt.axis('equal')  # 保证坐标轴比例一致
    plt.savefig("distance_visualization.png")
    plt.close()
    print("可视化结果已保存至 distance_visualization.png")

#
# # 示例使用
# file_path = r"D:\Desktop\车道.txt"
# map_features = parse_opendrive_map(file_path)
# if map_features is not None:
#     encoded_features = encode_map_features(map_features)
#     print(encoded_features)
def main(participant_file, lane_file):
    # 转换车道数据格式

    # 读取车道文件
    lanes = parse_opendrive_map(lane_file)
    if lanes is None:
        return
    kd_tree, lane_info_list, all_segments,segment_indices = preprocess_lane_geometry(lanes['lanes'])
    if kd_tree is None:
        print("No valid segments for KDTree, cannot continue processing.")
        return

    # 读取交通参与者文件
    frames = read_participant_file(participant_file)

    # 处理每一帧数据
    for frame_idx, participants in enumerate(frames):
        print(f"Processing frame {frame_idx}...")
        encoded_features = []
        closest_distances = []
        for participant in participants:
            closest_lane = get_closest_lane_features(participant[:2], kd_tree, lane_info_list, all_segments,segment_indices)
            closest_distances.append(closest_lane)
            if closest_lane is None:
                default_features = torch.FloatTensor([0] * 10)
                encoded_features.append(default_features)
                continue
            encoded = encode_participant_features(participant, closest_lane)
            encoded_features.append(encoded)
        if encoded_features:
            encoded_features = torch.stack(encoded_features)
            # print(f"Frame {frame_idx} processed. Features:")
            # print(encoded_features)  # 打印所有特征值
            # print(f"Shape: {encoded_features.shape}\n")
            # 每帧生成可视化
        if frame_idx == 0:  # 仅可视化第一帧用于调试
            visualize_scene(all_segments, participants, [d['distance'] if d else None for d in closest_distances])


if __name__ == "__main__":
    participant_file = r"D:\Desktop\output.csv"
    lane_file = r"D:\Github\Onsite_rule_driven_model-main\A\highway_merge_2_2_310\highway_merge_2_2_310.xodr"
    main(participant_file, lane_file)