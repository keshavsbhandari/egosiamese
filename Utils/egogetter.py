import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from Utils.utils import readFromPickle
from Extras.loadconfigs import DEPTH, SINGLE_VIDEO_POLICY, BALANCE_VIDEOS, TRAIN_FROM_PICKLE, SERVER, DATA_ROOT

class VideoGetter(object):
    def __init__(self, 
                 root, 
                 mode = 'train', 
                 nwor = DEPTH, 
                 shuffle = False, 
                 truncate = None):
        assert mode in ['train','test'], f"mode must be in [train, test]"
        self.root = Path(root)
        self.content = sorted(Path(root).glob(f'*/*/{mode}'))
        self.spatial = [c/"frames" for c in self.content]
        self.motion = [f/"flows" for f in self.content]
        self.nwor = nwor
        if TRAIN_FROM_PICKLE:
            self.frames_list = readFromPickle(f"Extras/{mode}.pkl")
        else:    
            if SINGLE_VIDEO_POLICY:    
                self.frames_list = self.getSingleFramesPerVideo()[:truncate]
            else:
                self.frames_list = self.getFramesList()[:truncate]
        
        if shuffle:
            random.shuffle(self.frames_list)
            
    def getSingleFramesPerVideo(self):
        frame_list = []
        for spatial in self.spatial:
            videos = sorted(spatial.glob('*'))
            if BALANCE_VIDEOS>0:
                videos = videos[:BALANCE_VIDEOS]
            for video in videos:
                frames = sorted(video.glob('*.jpg'))
                frames = frames[(len(frames)//2 - self.nwor//2):(len(frames)//2+self.nwor//2)]
                assert len(frames) == self.nwor, "Frames size doesn't match the depth"
                frame_list.append(frames)
        return frame_list
    
    def getFramesList(self):
        frames_list = []
        for spatial in self.spatial:
            videos = sorted(spatial.glob('*'))
            if BALANCE_VIDEOS>0:
                videos = videos[:BALANCE_VIDEOS]
            for video in videos:
                frames = sorted(video.glob('*.jpg'))
                n = len(frames)
                frames = (np.array(frames)[-n - n%self.nwor:][range(0, n - n%self.nwor, self.nwor)]).tolist()
                frames_list.extend(frames)
        return frames_list
    
    def __len__(self):
        return len(self.frames_list)
    
    def getFlowPath(self, path):
        return Path(path.as_posix().replace("frames","flows").replace(".jpg",".npy"))
    
    def getNextFlowFrame(self, path):
        frame_parent = path.parent
        next_file = f"{str(int(path.stem)+1).zfill(len(path.stem))}{path.suffix}"
        
        frame = frame_parent/next_file
        flow = Path(frame.as_posix().replace("frames","flows").replace(".jpg",".npy"))
        
        label = self.getLabel(frame)
        
        
        assert frame.exists(), f"{frame} doesn't exist"
        
        return frame, flow, label
        
    
    def getLabel(self, path):
        action = path.parent.parent.parent.parent
        activity = action.parent
        labels = {'s_label': activity.name.lower(), 'm_label':action.name.lower(), 'label':f"{activity.name.lower()}_{action.name.lower()}"}
        return labels
    
    def getSingleFrameSeqPerVideo(self, idx):
        frames = self.frames_list[idx]
        if SERVER == 'PANDAS':
            for i, frame in enumerate(frames):
                frame = Path(frame.as_posix().replace("/data/keshav/360/finalEgok360/data/", DATA_ROOT))
                frames[i] = frame
        data = self.getLabel(frames[0])
        data.update({'frame':frames, 'flow':[]})
        for frame in frames:
            _, flow, _ = self.getNextFlowFrame(frame)
            if not flow.exists():
                pre_flow = data['flow'][0]
                flow = pre_flow.parent/f"{str(int(pre_flow.stem) - 1).zfill(len(pre_flow.stem))}{pre_flow.suffix}"
                assert flow.exists(), f"{flow} doesn't exists"
                data['flow'].insert(0, flow)
            else:
                data['flow'].append(flow)
        return data
    
    def getMultiFrameSeqPerVideo(self, idx):
        frame = self.frames_list[idx]
        flow = self.getFlowPath(frame)
        data = self.getLabel(frame)
        
        assert frame.exists(), f"{frame} doesn't exist"
        assert flow.exists(), f"{flow} doesn't exist"
        
        data.update({'frame':[frame], 'flow':[flow]})
        
        for _ in range(1,self.nwor):
            frame, flow, label = self.getNextFlowFrame(frame)
            
            data['frame'].append(frame)
            
            if not flow.exists():
                pre_flow = data['flow'][0]
                flow = pre_flow.parent/f"{str(int(pre_flow.stem) - 1).zfill(len(pre_flow.stem))}{pre_flow.suffix}"
                assert flow.exists(), f"{flow} doesn't exists"
                
                data['flow'].insert(0, flow)
                
            else:
                data['flow'].append(flow)
        
        return data
    
    def __getitem__(self, idx):
        if TRAIN_FROM_PICKLE:
            return self.getSingleFrameSeqPerVideo(idx)
        if SINGLE_VIDEO_POLICY:
            return self.getSingleFrameSeqPerVideo(idx)
        else:
            return self.getMultiFrameSeqPerVideo(idx)
        
                


if __name__ == '__main__':
    root = "/data/keshav/360/finalEgok360/data/"
    vd = VideoGetter(root = "/data/keshav/360/finalEgok360/data/", mode = 'test', shuffle = True)
    
    for _ in tqdm(vd):pass
    