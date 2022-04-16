from Utils.egoloader import getEgoLoader
from tqdm import tqdm
from Utils.utils import stopExec

if __name__ == '__main__':
    loader = getEgoLoader(mode = 'train', 
                          shuffle = True, 
                          siamese = False, 
                          focus = False, 
                          flo = False, 
                          img = True, 
                          low = -1, 
                          high = 1, 
                          double_rots = False
                          )
    
    for _ in tqdm(loader):
        stopExec()