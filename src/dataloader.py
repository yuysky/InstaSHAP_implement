import os
import numpy as np
import csv
import gzip

def preprocess_bike_sharing_dataset(load_dataset_path):

    csv_path = os.path.join(load_dataset_path, 'hour.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found at: {csv_path}. Please check your path.")
        
    print(f"Processing file: {csv_path}")
    
    with open(csv_path, 'r') as file:
        lines = file.readlines()
    
    header = lines[0]
    data_lines = lines[1:]
    N = len(data_lines)
    bike_share_data = np.zeros((N, 17))

    for i, line in enumerate(data_lines):
        values = line.strip().split(',')
        # modify date string to just day number if needed, though logic here takes chars 8:10
        try:
            day = int(values[1][8:10])
            values[1] = day
        except:
            values[1] = 0 # Fallback

        bike_share_data[i] = [float(val) for val in values]
        
    # 1. Drop the first column (instant ID) -> 16 columns left
    bike_share_data = bike_share_data[:, 1:] 
    
    # 2. Log transform specific columns (casual, registered, cnt)
    # Indices 13, 14, 15 correspond to casual, registered, cnt in the remaining 16 cols
    bike_share_data[:, 13:16] = np.log(1 + bike_share_data[:, 13:16])
    
    # 3. extract X (Features) and Y (Target)
    X = bike_share_data[:, :13]  # Features
    Y = bike_share_data[:, 15]   # Target: cnt (total count)
    
    XY_stuff = (X, Y, None, False, None)

    # Labels 
    readable_labels = {
        0 : "day", 1 : "season", 2 : "year", 3 : "month", 4 : "hour",
        5 : "holiday", 6 : "day of week", 7 : "workday",
        8 : "weather", 9 : "temperature", 10 : "feels_like_temp",
        11 : "humidity", 12 : "wind speed",
    }
    
    full_readable_labels = {
        'task_type' : "regression",
    }
    
    label_stuff = (readable_labels, full_readable_labels)
    # print('--- processed and NOT saved ---')

    return XY_stuff, label_stuff

class CustomBikeDataset:
    def __init__(self, root_dir="../data/", dataset_name="bike_sharing", seed=None):

        self.dataset_path = os.path.join(root_dir, dataset_name)
        
        XY_stuff, label_stuff = preprocess_bike_sharing_dataset(self.dataset_path)
        
        raw_X, raw_Y = XY_stuff[0], XY_stuff[1]

        if len(raw_Y.shape) == 1:
            raw_Y = raw_Y[:, None]
        
        total_len = raw_X.shape[0]
        # Fixed 80% split for Train+Val / Test
        test_split_idx = int(total_len * 0.8)

        self.trnvalX = raw_X[:test_split_idx]
        self.trnvalY = raw_Y[:test_split_idx]
        self.tstX = raw_X[test_split_idx:]
        self.tstY = raw_Y[test_split_idx:]

        self.trnX, self.valX, self.trnY, self.valY = None, None, None, None

        self.readable_labels = label_stuff[0]
        
        self.label_stuff_dict = {
            "readable_labels": self.readable_labels,
            "full_readable_labels": {
                "task_type": "regression",
                "D0": self.trnvalX.shape[1]
            }
        }
        
        if seed is not None:
            self.shuffle_and_split_trnval(trnval_shuffle_seed=seed)

    def shuffle_and_split_trnval(self, trnval_shuffle_seed=None, trnval_split_percentage=0.7):

        if trnval_shuffle_seed is None:
            np.random.seed(None)
            self.trnval_shuffle_seed = np.random.randint(0, 10000)
        else:
            self.trnval_shuffle_seed = trnval_shuffle_seed
            
        np.random.seed(self.trnval_shuffle_seed)
        print(f'Splitting Train/Val with seed: {self.trnval_shuffle_seed}')

        M_NUM = self.trnvalX.shape[0]
        rand_indices = np.random.permutation(M_NUM)
        M_TRN_NUM = int(M_NUM * trnval_split_percentage)
        
        self.trnX = self.trnvalX[rand_indices[:M_TRN_NUM]]
        self.valX = self.trnvalX[rand_indices[M_TRN_NUM:]]
        self.trnY = self.trnvalY[rand_indices[:M_TRN_NUM]]
        self.valY = self.trnvalY[rand_indices[M_TRN_NUM:]]

    def pull_data(self):

        return (self.trnvalX, self.trnvalY, self.tstX, self.tstY)

    def pull_trnval_data(self):

        if self.trnX is None:
            # Auto split if not done yet
            self.shuffle_and_split_trnval()
        return (self.trnX, self.trnY, self.valX, self.valY)

    def get_D(self):
        return self.trnvalX.shape[1]
        
    def get_C(self):
        return self.trnvalY.shape[1]

    def get_dataset_id(self):
        return "bike_sharing_custom"
        
    def get_readable_labels(self):
        return self.label_stuff_dict["readable_labels"]
        
    def get_full_readable_labels(self):
        return self.label_stuff_dict["full_readable_labels"]

    def get_task_type(self):
        return self.get_full_readable_labels()["task_type"]

    def get_grouped_feature_dict(self):
        D = self.get_D()
        
        grouped_features_dict = {}
        grouped_features_dict["D"] = D
        grouped_features_dict["D0"] = D
        
        for i in range(D):
            grouped_features_dict[i] = [i]
            
        return grouped_features_dict


# if __name__ == "__main__":

#     try:
#         print("Initializing Custom Dataset...")
#         dataset_obj = CustomBikeDataset(root_dir="../data/", dataset_name="bike_sharing", seed=37)
#         dataset_obj.shuffle_and_split_trnval()
        
#         trnX, trnY, valX, valY = dataset_obj.pull_trnval_data()
#         print(f"Dataset ready.")
#         print(f"Train X: {trnX.shape}, Y: {trnY.shape}")
#         print(f"Val   X: {valX.shape}, Y: {valY.shape}")
#         print(f"Test  X: {dataset_obj.tstX.shape}, Y: {dataset_obj.tstY.shape}")
#     except Exception as e:
#         print(f"Test failed: {e}")
#         print("Please ensure '../data/bike_sharing/hour.csv' exists relative to this script.")


def preprocess_tree_cover_dataset(load_dataset_path):
    csv_path = os.path.join(load_dataset_path, "covtype.data.gz")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found at: {csv_path}")

    # Dataset size constants
    N = 581012
    all_data_array = np.zeros((N, 55), dtype=int)

    with gzip.open(csv_path, mode="rt", newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(reader):
            all_data_array[i] = [int(val) for val in row]

    # Remap 40 soil binary columns to 4 categories
    soil_remappings = { 
        0 : list(range(0, 6)),   # lower montane
        1 : list(range(6, 18)),  # upper montane
        2 : list(range(18, 34)), # subalpine
        3 : list(range(34, 40)), # alpine  
    }
    soil_remapping_tensor = np.zeros((40, 4))
    for c2 in range(4):
        for c1 in soil_remappings[c2]:
            soil_remapping_tensor[c1, c2] = 1

    all_simple_soils = np.matmul(all_data_array[:, 14:54], soil_remapping_tensor)
    # Resulting X: [10 quant] + [4 wilderness] + [4 simple soil] = 18 features
    X = np.concatenate([all_data_array[:, :14], all_simple_soils], axis=1).astype(np.float32)
    
    # Target: Convert 1-7 range to 0-6 for CrossEntropyLoss
    Y = (all_data_array[:, 54] - 1).astype(int) 

    XY_stuff = (X, Y, None, True, 0)
    
    full_readable_labels = {
        "task_type": "multiclass_classification",
        "D0": 12, # Original grouped feature count
        0: {"label": "elevation (m)"},
        1: {"label": "aspect (azimuth)"},
        2: {"label": "slope (deg)"},
        3: {"label": "Horizontal_Distance_To_Hydrology"},
        4: {"label": "Vertical_Distance_To_Hydrology"},
        5: {"label": "Horizontal_Distance_To_Roadways"},
        6: {"label": "Hillshade_9am"},
        7: {"label": "Hillshade_Noon"},
        8: {"label": "Hillshade_3pm"},
        9: {"label": "Horizontal_Distance_To_Fire_Points"},
        10: {"label": "wilderness area"},
        11: {"label": "soil type"},
    }
    
    readable_labels = {k: v["label"] for k, v in full_readable_labels.items() if isinstance(k, int)}
    return XY_stuff, (readable_labels, full_readable_labels)


class CustomTreeDataset:
    def __init__(self, root_dir="../data/", dataset_name="covertype", seed=None):
        self.dataset_path = os.path.join(root_dir, dataset_name)
        XY_stuff, label_stuff = preprocess_tree_cover_dataset(self.dataset_path)
        
        raw_X, raw_Y = XY_stuff[0], XY_stuff[1]
        
        # Classification targets usually stay as (N,) for CrossEntropy
        total_len = raw_X.shape[0]
        test_split_idx = int(total_len * 0.8)

        self.trnvalX, self.trnvalY = raw_X[:test_split_idx], raw_Y[:test_split_idx]
        self.tstX, self.tstY = raw_X[test_split_idx:], raw_Y[test_split_idx:]
        
        self.trnX, self.valX, self.trnY, self.valY = None, None, None, None
        self.readable_labels = label_stuff[0]
        self.label_stuff_dict = {
            "readable_labels": self.readable_labels,
            "full_readable_labels": {
                "task_type": "multiclass_classification",
            }
        }
        
        if seed is not None:
            self.shuffle_and_split_trnval(trnval_shuffle_seed=seed)
        
        if len(self.trnvalY.shape) == 1:
            self.trnvalY = self.trnvalY[:, np.newaxis]
        if len(self.tstY.shape) == 1:
            self.tstY = self.tstY[:, np.newaxis]

    def shuffle_and_split_trnval(self, trnval_shuffle_seed=None, trnval_split_percentage=0.7):
        if trnval_shuffle_seed is None:
            self.trnval_shuffle_seed = np.random.randint(0, 10000)
        else:
            self.trnval_shuffle_seed = trnval_shuffle_seed
            
        np.random.seed(self.trnval_shuffle_seed)
        M_NUM = self.trnvalX.shape[0]
        rand_indices = np.random.permutation(M_NUM)
        M_TRN_NUM = int(M_NUM * trnval_split_percentage)
        
        self.trnX = self.trnvalX[rand_indices[:M_TRN_NUM]]
        self.valX = self.trnvalX[rand_indices[M_TRN_NUM:]]
        self.trnY = self.trnvalY[rand_indices[:M_TRN_NUM]]
        self.valY = self.trnvalY[rand_indices[M_TRN_NUM:]]

    def pull_data(self):

        return (self.trnvalX, self.trnvalY, self.tstX, self.tstY)

    def pull_trnval_data(self):

        if self.trnX is None:
            # Auto split if not done yet
            self.shuffle_and_split_trnval()
        return (self.trnX, self.trnY, self.valX, self.valY)

    def get_D(self):
        return self.trnvalX.shape[1]
        
    def get_C(self):
        # Returns number of classes
        return int(np.max(self.trnvalY) + 1)

    def get_grouped_feature_dict(self):
        D = self.get_D()
        return {**{"D": D, "D0": D}, **{i: [i] for i in range(D)}}
    
    def get_dataset_id(self):
        return "tree_over_custom"

    def get_readable_labels(self):
        return self.label_stuff_dict["readable_labels"]
        
    def get_full_readable_labels(self):
        return self.label_stuff_dict["full_readable_labels"]

    def get_task_type(self):
        return self.get_full_readable_labels()["task_type"]