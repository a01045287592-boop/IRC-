"""Optional PyTorch loader. This is NOT a policy training or deployment program."""
from pathlib import Path
import numpy as np
from review import chunk

class Tray4Dataset:
    """Split parent episode folders first; never randomly split this sample index."""
    def __init__(self,episode_dirs,horizon=30):
        self.horizon=horizon;self.episodes=[];self.samples=[]
        for folder in map(Path,episode_dirs):
            data=np.load(folder/'training/episode.npz',allow_pickle=False)
            idx=len(self.episodes);self.episodes.append((folder,data))
            self.samples.extend((idx,i) for i in range(len(data['state'])))
    def __len__(self):return len(self.samples)
    def __getitem__(self,index):
        import torch
        from PIL import Image
        ep,t=self.samples[index];folder,data=self.episodes[ep]
        image=np.asarray(Image.open(folder/str(data['image'][t])).convert('RGB'),dtype=np.float32)/255
        action,is_pad=chunk(data,t,self.horizon)
        return {'observation.images.front':torch.from_numpy(image.transpose(2,0,1).copy()),
                'observation.state':torch.from_numpy(data['state'][t].copy()),
                'action':torch.from_numpy(action), 'action_is_pad':torch.from_numpy(is_pad)}
