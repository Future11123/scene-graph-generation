import logging
import os
import pickle
import random

import numpy as np


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)


class DataLoader:
    def __init__(
        self, batch_size=8, seq_length=10, forcePreProcess=True, infer=False
    ):
        """
        Initialiser function for the DataLoader class
        params:
        batch_size : Size of the mini-batch
        seq_length : Sequence length to be considered  
        datasets : The indices of the datasets to use
        forcePreProcess : Flag to forcefully preprocess the data again from csv files
        """
        # random.seed(42)
        # np.random.seed(42)
        # List of data directories where raw data resides
        self.data_dirs = "../data/prediction_train/"
        self.dataset_cnt = len(os.listdir(self.data_dirs))
        self.dataset_idx = sorted(os.listdir(self.data_dirs))
        np.random.shuffle(self.dataset_idx)
        self.train_data_dirs = self.dataset_idx[: int(self.dataset_cnt * 0.9)]
        if infer == True:
            self.train_data_dirs = self.dataset_idx[int(self.dataset_cnt * 0.9) :]
        self.infer = infer

        # Store the arguments
        self.batch_size = batch_size
        self.seq_length = seq_length

        data_file = os.path.join("../data/", "trajectories.cpkl")
        if infer == True:
            data_file = os.path.join("../data/", "test_trajectories.cpkl")

        self.val_fraction = 0.2

        # If the file doesn't exist or forcePreProcess is true
        if not (os.path.exists(data_file)) or forcePreProcess:
            print("Creating pre-processed data from raw data")
            # Preprocess the data from the csv files of the datasets
            # Note that this data is processed in frames
            self.frame_preprocess(self.train_data_dirs, data_file)

        # Load the processed data from the pickle file
        self.load_preprocessed(data_file)
        # Reset all the data pointers of the dataloader object
        self.reset_batch_pointer(valid=False)
        self.reset_batch_pointer(valid=True)

    # def class_objtype(self, object_type):
    #     if object_type == 1 or object_type == 2:
    #         return 3
    #     elif object_type == 3:
    #         return 1
    #     elif object_type == 4:
    #         return 2
    #     else:
    #         return -1

    def frame_preprocess(self, data_dirs, data_file):
        """
        Function that will pre-process the pixel_pos.csv files of each dataset
        into data with occupancy grid that can be used
        params:
        data_dirs : List of directories where raw data resides
        data_file : The file into which all the pre-processed data needs to be stored
        """
        # all_frame_data would be a list of list of numpy arrays corresponding to each dataset
        # Each numpy array will correspond to a frame and would be of size (numPeds, 3) each row
        # containing pedID, x, y
        all_frame_data = []#每个元素是一个列表，对应一个文件，包含每一帧的数据矩阵
        # Validation frame data
        valid_frame_data = []
        # frameList_data would be a list of lists corresponding to each dataset
        # Each list would contain the frameIds of all the frames in the dataset
        frameList_data = []#每个元素是一个列表，对应一个文件，装着这个文件里面所有帧的ID
        # numPeds_data would be a list of lists corresponding to each dataset
        # Each list would contain the number of pedestrians in each frame in the dataset
        numPeds_data = []#每个元素是一个列表，对应一个文件，装着这个文件里面每一帧里agent的数量
        # Index of the current dataset
        dataset_index = 0

        # 修改为10000
        min_position_x = 10000
        max_position_x = -10000
        min_position_y = 10000
        max_position_y = -10000


        for ind_directory, directory in enumerate(data_dirs):
            file_path = os.path.join("../data/prediction_train/", directory)
            data = np.genfromtxt(file_path, delimiter=" ")
#            print(directory)
            min_position_x = min(min_position_x, min(data[:, 2]))
            max_position_x = max(max_position_x, max(data[:, 2]))
            min_position_y = min(min_position_y, min(data[:, 3]))
            max_position_y = max(max_position_y, max(data[:, 3]))

        # For each dataset
        for ind_directory, directory in enumerate(data_dirs):
            # define path of the csv file of the current dataset
            # file_path = os.path.join(directory, 'pixel_pos.csv')

            file_path = os.path.join("../data/prediction_train/", directory)

            # Load the data from the csv file
            data = np.genfromtxt(file_path, delimiter=" ")
            """
            data[:, 3] = (
                (data[:, 3] - min(data[:, 3])) / (max(data[:, 3]) - min(data[:, 3]))
            ) * 2 - 1
            data[:, 4] = (
                (data[:, 4] - min(data[:, 4])) / (max(data[:, 4]) - min(data[:, 4]))
            ) * 2 - 1
            """
            data[:, 2] = (
                (data[:, 2] - min_position_x) / (max_position_x - min_position_x)
            ) * 2 - 1
            data[:, 3] = (
                (data[:, 3] - min_position_y) / (max_position_y - min_position_y)
            ) * 2 - 1

            # data = data[~(data[:, 2] == 5)]#~是按二进制位取反运算,这一步把第三列是5的行全部删掉了

            # Frame IDs of the frames in the current dataset（frameList是一个列表，含有一个文件里所有frame的编号）
            frameList = np.unique(data[:, 0]).tolist()#np.unique()是除去数组中重复数字并排序输出,np.tolist()是把数组或矩阵转换为列表
            numFrames = len(frameList)

            # Add the list of frameIDs to the frameList_data
            frameList_data.append(frameList)
            #对应每个frameList加个相应的空列表，在下面的for循环里面填数据进去
            # Initialize the list of numPeds for the current dataset
            numPeds_data.append([])
            # Initialize the list of numpy arrays for the current dataset
            all_frame_data.append([])
            # Initialize the list of numpy arrays for the current dataset
            valid_frame_data.append([])

            skip = 1

            #这是一个文件之内的for循环
            for ind, frame in enumerate(frameList):

                ## NOTE CHANGE
                if ind % skip != 0:
                    # Skip every n frames
                    continue

                # Extract all pedestrians in current frame
                pedsInFrame = data[data[:, 0] == frame, :]

                # Extract peds list
                pedsList = pedsInFrame[:, 1].tolist()

                # Add number of peds in the current frame to the stored data
                numPeds_data[dataset_index].append(len(pedsList))

                # Initialize the row of the numpy array
                pedsWithPos = []
                # For each ped in the current frame
                #这是一帧之内的for循环
                for ped in pedsList:
                    # Extract their x and y positions 所有特征
                    current_x = pedsInFrame[pedsInFrame[:, 1] == ped, 2][0]#[0]是为了把只有一个数的[1,1]矩阵变成一个数
                    current_y = pedsInFrame[pedsInFrame[:, 1] == ped, 3][0]
                    heading = pedsInFrame[pedsInFrame[:, 1] == ped, 4][0]
                    current_type = pedsInFrame[pedsInFrame[:, 1] == ped, 5][0]
                    lane_type = pedsInFrame[pedsInFrame[:, 1] == ped, 6][0]
                    lane_width = pedsInFrame[pedsInFrame[:, 1] == ped, 7][0]
                    road_mark_type = pedsInFrame[pedsInFrame[:, 1] == ped, 8][0]
                    distance = pedsInFrame[pedsInFrame[:, 1] == ped, 11][0]

                    # current_inter = pedsInFrame[pedsInFrame[:, 1] == ped, 5][0] - 1
                    # print('current_type    {}'.format(current_type))
                    # Add their pedID, x, y to the row of the numpy array
                    #pedsWithPos是一个列表，每个元素是一帧里面每个agent的[ID，x, y, type, inter]

                    pedsWithPos.append([ped, current_x, current_y, heading, current_type, lane_type, lane_width, road_mark_type, distance])

                if (ind > numFrames * self.val_fraction) or (self.infer):
                    # At inference time, no validation data
                    # Add the details of all the peds in the current frame to all_frame_data
                    all_frame_data[dataset_index].append(
                        np.array(pedsWithPos)
                    )  # different frame (may) have diffenent number person
                else:
                    valid_frame_data[dataset_index].append(np.array(pedsWithPos))

            dataset_index += 1
        # Save the tuple (all_frame_data, frameList_data, numPeds_data) in the pickle file
        f = open(data_file, "wb")
        pickle.dump(
            (all_frame_data, frameList_data, numPeds_data, valid_frame_data),
            f,
            protocol=2,
        )
        f.close()

    def load_preprocessed(self, data_file):
        """
        Function to load the pre-processed data into the DataLoader object
        params:
        data_file : the path to the pickled data file
        """
        # Load data from the pickled file
        f = open(data_file, "rb")
        self.raw_data = pickle.load(f)
        f.close()
        # Get all the data from the pickle file
        self.data = self.raw_data[0]
        self.frameList = self.raw_data[1]
        self.numPedsList = self.raw_data[2]
        self.valid_data = self.raw_data[3]
        counter = 0
        valid_counter = 0

        # For each dataset
        for dataset in range(len(self.data)):
            # get the frame data for the current dataset
            all_frame_data = self.data[dataset]
            valid_frame_data = self.valid_data[dataset]
            print(
                "Training data from dataset {} : {}".format(
                    dataset, len(all_frame_data)
                )
            )
            print(
                "Validation data from dataset {} : {}".format(
                    dataset, len(valid_frame_data)
                )
            )
            # Increment the counter with the number of sequences in the current dataset
            counter += int(len(all_frame_data) / (self.seq_length))#算所有数据一共有多少个样本（帧数达到seq_length为一个时间序列样本）余数会直接舍去
            valid_counter += int(len(valid_frame_data) / (self.seq_length))

        # Calculate the number of batches
        self.num_batches = int(counter / self.batch_size)
        self.valid_num_batches = int(valid_counter / self.batch_size)
        print("Total number of training batches: {}".format(self.num_batches * 2))
        print("Total number of validation batches: {}".format(self.valid_num_batches))
        # On an average, we need twice the number of batches to cover the data
        # due to randomization introduced
        self.num_batches = self.num_batches * 2
        # self.valid_num_batches = self.valid_num_batches * 2

    def next_batch(self, randomUpdate=True):#获得每个batch，包括input和标签，需要修改
        """
        Function to get the next batch of points
        """
        # Source data
        x_batch = []#一个列表，有batch_size个元素，每个元素对应一个样本，一个样本是seq_length帧数据
        # Target data
        y_batch = []
        # Frame data
        frame_batch = []#一个列表，有batch_size个元素，每个元素对应一个样本，含有对应每一帧的帧ID
        # Dataset data
        d = []#一个列表，有batch_size个元素，每个元素对应一个样本，指出一个样本来自哪个dataset，即来自哪个文件
        # Iteration index
        i = 0

        while i < self.batch_size:
            # Extract the frame data of the current dataset

            frame_data = self.data[self.dataset_pointer]
            frame_ids = self.frameList[self.dataset_pointer]
            # Get the frame pointer for the current dataset
            idx = self.frame_pointer
            # While there is still seq_length number of frames left in the current dataset

            if idx + self.seq_length < len(frame_data):
                # All the data in this sequence
                # seq_frame_data = frame_data[idx:idx+self.seq_length+1]
                seq_source_frame_data = frame_data[idx : idx + self.seq_length]
                seq_target_frame_data = frame_data[idx + 1 : idx + self.seq_length + 1]
                seq_frame_ids = frame_ids[idx : idx + self.seq_length]

                # Number of unique peds in this sequence of frames
                x_batch.append(seq_source_frame_data)
                y_batch.append(seq_target_frame_data)
                frame_batch.append(seq_frame_ids)

                # advance the frame pointer to a random point
                if randomUpdate:
                    self.frame_pointer += random.randint(1, self.seq_length)
                else:
                    self.frame_pointer += self.seq_length

                d.append(self.dataset_pointer)
                i += 1

            else:
                # Not enough frames left
                # Increment the dataset pointer and set the frame_pointer to zero
                self.tick_batch_pointer(valid=False)

        return x_batch, y_batch, frame_batch, d#x_batch是列表，每个元素是一个包含seq个数矩阵的列表，每个矩阵是一帧的数据，但矩阵的行数不一定相同

    def next_valid_batch(self, randomUpdate=True):
        """
        Function to get the next Validation batch of points
        """
        # Source data
        x_batch = []
        # Target data
        y_batch = []
        # Dataset data
        d = []
        # Iteration index
        i = 0
        while i < self.batch_size:
            # Extract the frame data of the current dataset
            frame_data = self.valid_data[self.valid_dataset_pointer]
            # Get the frame pointer for the current dataset
            idx = self.valid_frame_pointer
            # While there is still seq_length number of frames left in the current dataset
            if idx + self.seq_length < len(frame_data):
                # All the data in this sequence
                # seq_frame_data = frame_data[idx:idx+self.seq_length+1]
                seq_source_frame_data = frame_data[idx : idx + self.seq_length]
                seq_target_frame_data = frame_data[idx + 1 : idx + self.seq_length + 1]

                # Number of unique peds in this sequence of frames
                x_batch.append(seq_source_frame_data)
                y_batch.append(seq_target_frame_data)

                # advance the frame pointer to a random point
                if randomUpdate:
                    self.valid_frame_pointer += random.randint(1, self.seq_length)
                else:
                    self.valid_frame_pointer += self.seq_length

                d.append(self.valid_dataset_pointer)
                i += 1

            else:
                # Not enough frames left
                # Increment the dataset pointer and set the frame_pointer to zero
                self.tick_batch_pointer(valid=True)

        return x_batch, y_batch, d

    def tick_batch_pointer(self, valid=False):
        """
        Advance the dataset pointer
        """
        if not valid:
            # Go to the next dataset
            self.dataset_pointer += 1
            # Set the frame pointer to zero for the current dataset
            self.frame_pointer = 0
            # If all datasets are done, then go to the first one again
            if self.dataset_pointer >= len(self.data):
                self.dataset_pointer = 0
        else:
            # Go to the next dataset
            self.valid_dataset_pointer += 1
            # Set the frame pointer to zero for the current dataset
            self.valid_frame_pointer = 0
            # If all datasets are done, then go to the first one again
            if self.valid_dataset_pointer >= len(self.valid_data):
                self.valid_dataset_pointer = 0

    def reset_batch_pointer(self, valid=False):
        """
        Reset all pointers
        """
        if not valid:
            # Go to the first frame of the first dataset
            self.dataset_pointer = 0
            self.frame_pointer = 0
        else:
            self.valid_dataset_pointer = 0
            self.valid_frame_pointer = 0
