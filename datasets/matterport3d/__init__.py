import json
import glob

config_path = '/mnt/SSD4T/yiqinzhao/Xihe/matterport_configs'
matterport3d_root = '/mnt/IRONWOLF1/yiqinzhao/RawDatasets/Matterport3D/v1/scans'


class Matterport3DList:
    def __init__(self):
        self.configs = {}

    def get_config(self, scene_id, depth_name):
        if scene_id not in self.configs:
            self.configs[scene_id] = json.load(
                open(f'{config_path}/{scene_id}.json'))

        return self.configs[scene_id][f'{depth_name}.png']
