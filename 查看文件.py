import csv
import time
import xml.etree.ElementTree as ET
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from scipy.spatial import KDTree
from matplotlib.patches import Polygon, Rectangle
from scipy.interpolate import interp1d

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
            'traffic_signals': [],
            'speed_limits': [],
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
                        # 处理曲线几何元素
                        curve = geom.find('arc')
                        if curve is not None:
                            curvature = float(curve.get('curvature', 0.0))
                            num_points = 50  # 插值点数
                            s_vals = np.linspace(0, length, num_points)
                            x_vals = []
                            y_vals = []
                            for s_val in s_vals:
                                if curvature == 0:
                                    x_val = x + s_val * np.cos(hdg)
                                    y_val = y + s_val * np.sin(hdg)
                                else:
                                    radius = 1 / curvature
                                    angle = s_val / radius
                                    x_val = x + radius * (np.sin(hdg + angle) - np.sin(hdg))
                                    y_val = y - radius * (np.cos(hdg + angle) - np.cos(hdg))
                                x_vals.append(x_val)
                                y_vals.append(y_val)
                            road_geometry.extend([{
                                's': s + s_val,
                                'x': x_val,
                                'y': y_val,
                                'hdg': hdg,
                                'length': length,
                                'type': geom.tag
                            } for s_val, x_val, y_val in zip(s_vals, x_vals, y_vals)])
                        else:
                            # 计算终点坐标（根据hdg和length）
                            end_x = x + length * np.cos(hdg)
                            end_y = y + length * np.sin(hdg)
                            road_geometry.append({
                                's': s,
                                'x': x,
                                'y': y,
                                'end_x': end_x,
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
                            width_elements = lane.findall('width')
                            if not width_elements:
                                continue  # 如果没有width这个标签说明为地面标线，不是车道
                            width_value = 0
                            for width in width_elements:
                                a = float(width.get('a', 0.0))
                                b = float(width.get('b', 0.0))
                                c = float(width.get('c', 0.0))
                                d = float(width.get('d', 0.0))
                                offset_pre = float(width.get('sOffset', 0.0))
                                # 这里简单取第一个width标签的值作为车道宽度，可根据实际情况修改
                                width_value = a
                            if width_value == 0:
                                # 尝试通过前一个车道数据来推算宽度
                                lane_index = list(dir_section).index(lane)
                                if lane_index > 0:
                                    prev_lane = list(dir_section)[lane_index - 1]
                                    prev_width_elements = prev_lane.findall('width')
                                    if prev_width_elements:
                                        prev_width = prev_width_elements[0]
                                        prev_a = float(prev_width.get('a', 0.0))
                                        width_value = prev_a
                                if width_value == 0:
                                    width_value = 3.5  # 如果推算失败，默认宽度为3.5

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


def preprocess_lane_geometry(lanes, segment_length=0.5):
    """
    预处理车道几何数据为可查询的结构
    :param lanes: 车道信息列表
    :param segment_length: 划分线段的长度，默认为 1.0
    :return: KDTree、车道信息列表、所有线段列表
    """
    all_segments = []
    lane_info = []

    for lane in lanes:
        geom = lane['geometry']
        for i in range(len(geom) - 1):
            try:
                start = np.array([geom[i]['x'], geom[i]['y']])
                end = np.array([geom[i + 1]['x'], geom[i + 1]['y']])
                # 计算两点之间的距离
                distance = np.linalg.norm(end - start)
                # 计算需要划分的段数
                num_segments = int(np.ceil(distance / segment_length))
                for j in range(num_segments):
                    current_start = start + j * (end - start) / num_segments
                    if j == num_segments - 1:
                        current_end = end
                    else:
                        current_end = start + (j + 1) * (end - start) / num_segments
                    all_segments.append((tuple(current_start), tuple(current_end)))
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
        return None, None, None, None

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


def get_closest_lane_features(participant_pos, kdtree, lane_info, all_segments, segment_indices):
    """
    获取最近车道的特征

    :param participant_pos: 交通参与者的位置 (x, y)
    :param kdtree: 预构建的KDTree
    :param lane_info: 车道信息列表
    :param all_segments: 所有线段列表
    :return: 包含最近车道特征的字典或None
    """
    dist, idx = kdtree.query(participant_pos, distance_upper_bound=50)  # 扩大搜索范围
    if dist == np.inf:
        return None

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
    encoded_signals = [encode_traffic_signal(sig) for sig in map_features['traffic_signals']]
    encoded_speed_limits = [encode_speed_limit(limit) for limit in map_features['speed_limits']]
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
    frames = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行
        for row in reader:
            frame = int(row[0])
            x = float(row[2])
            y = float(row[3])
            heading = float(row[4])
            type_id = int(row[5])
            # 动态调整 frames 列表的长度
            while len(frames) <= frame:
                frames.append([])
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



def visualize_scene(all_segments, lane_info_list, participants, closest_info):
    """可视化场景（精确显示车道宽度和车辆尺寸）"""
    plt.figure(figsize=(12, 12))
    ax = plt.gca()

    # 1. 绘制车道区域（带真实宽度）
    for idx, seg in enumerate(all_segments):
        lane_info = lane_info_list[idx]
        width = lane_info['width']
        start, end = np.array(seg[0]), np.array(seg[1])

        # 计算线段方向向量
        direction = end - start
        length = np.linalg.norm(direction)
        if length == 0:
            continue
        unit_direction = direction / length

        # 计算垂直向量（用于宽度扩展）
        perpendicular = np.array([-unit_direction[1], unit_direction[0]]) * (width / 2)

        # 生成车道边界多边形
        left_boundary = [
            start + perpendicular,
            end + perpendicular,
            end - perpendicular,
            start - perpendicular
        ]

        # 绘制填充车道区域
        poly = Polygon(
            np.vstack(left_boundary),
            closed=True,
            edgecolor='none',
            facecolor='skyblue',
            alpha=0.5,
            zorder=1
        )
        ax.add_patch(poly)

        # 绘制车道中心线（可选）
        ax.plot([start[0], end[0]], [start[1], end[1]],
                color='navy',
                linewidth=0.5,
                linestyle='--',
                zorder=2)

    # 2. 绘制交通参与者（精确尺寸）
    for i, (participant, lane_info) in enumerate(zip(participants, closest_info)):
        x, y, heading_rad, _ = participant
        car_length = 4.0
        car_width = 2.0

        # 创建车辆矩形（中心坐标转左下角）
        rect = Rectangle(
            (x - car_length / 2, y - car_width / 2),
            car_length,
            car_width,
            edgecolor='red',
            facecolor='none',
            linewidth=1.5,
            zorder=3
        )

        # 应用旋转
        t = Affine2D().rotate_around(x, y, heading_rad) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)

        # 绘制最近距离指示线
        if lane_info is not None:
            closest_point = lane_info['closest_point']
            ax.plot([x, closest_point[0]], [y, closest_point[1]],
                    color='limegreen',
                    linestyle=':',
                    linewidth=1.5,
                    zorder=4)
            ax.annotate(
                f'd={lane_info["distance"]:.2f}m',
                xy=((x + closest_point[0]) / 2, (y + closest_point[1]) / 2),
                xytext=(5, 0),
                textcoords='offset points',
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.5)
            )

    # 3. 设置坐标轴和显示参数
    plt.title("Lane Width & Vehicle Size Visualization (1:1 Scale)")
    plt.xlabel("X Coordinate (meters)")
    plt.ylabel("Y Coordinate (meters)")
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.axis('equal')
    plt.savefig("lane_width_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("可视化结果已保存至 lane_width_visualization.png")

#
# # 示例使用
# file_path = r"D:\Desktop\车道.txt"
# map_features = parse_opendrive_map(file_path)
# if map_features is not None:
#     encoded_features = encode_map_features(map_features)
#     print(encoded_features)
def main(participant_file, lane_file):
    start_time = time.time()  # 记录开始时间
    # 转换车道数据格式
    participant_frames = read_participant_file(participant_file)
    num_frames = len(participant_frames)

    # 读取车道文件
    lanes = parse_opendrive_map(lane_file)
    if lanes is None:
        return
    kd_tree, lane_info_list, all_segments, segment_indices = preprocess_lane_geometry(lanes['lanes'])
    if kd_tree is None:
        print("No valid segments for KDTree, cannot continue processing.")
        return

    # 读取交通参与者文件
    frames = read_participant_file(participant_file)

    # 打开文件以保存编码特征
    with open('current.txt', 'w') as f:
        # 处理每一帧数据
        for frame_idx, participants in enumerate(frames):
            print(f"Processing frame {frame_idx}...")
            encoded_features = []
            closest_distances = []
            for agent_idx, participant in enumerate(participants):
                closest_lane = get_closest_lane_features(participant[:2], kd_tree, lane_info_list, all_segments, segment_indices)
                closest_distances.append(closest_lane)
                if closest_lane is None:
                    default_features = torch.FloatTensor([0] * 10)
                    encoded_features.append(default_features)
                else:
                    encoded = encode_participant_features(participant, closest_lane)
                    encoded_features.append(encoded)

            if encoded_features:
                encoded_features = torch.stack(encoded_features)
                # print(f"Frame {frame_idx} processed. Features:")
                # print(encoded_features)  # 打印所有特征值
                # print(f"Shape: {encoded_features.shape}\n")

                # 将编码特征按指定格式保存到文件
                for agent_idx, feature in enumerate(encoded_features):
                    row = [frame_idx, agent_idx] + feature.tolist()
                    row_str = " ".join(map(str, row))
                    f.write(row_str + "\n")

            # 每帧生成可视化
            # if frame_idx == 116:  # 仅可视化第116帧用于调试
            #     visualize_scene(all_segments, lane_info_list, participants, closest_distances)

    end_time = time.time()  # 记录结束时间
    total_time = end_time - start_time  # 计算总处理时间
    print(f"Total processing time: {total_time} seconds")


if __name__ == "__main__":
    participant_file = r"D:\Desktop\output.csv"
    lane_file = r"D:\Github\Onsite_rule_driven_model-main\sample\scene\mixed_952_32_1\mixed_952_32_1.xodr"
    main(participant_file, lane_file)