from Model.vitunified import UnifiedStream
from Utils.egoloader import getEgoLoader
import torch
if __name__ == '__main__':
    
    loader = getEgoLoader(mode = 'train', 
                          shuffle = True, 
                          siamese = False, 
                          focus = False, 
                          flo = True, 
                          img = True, 
                          low = -1, 
                          high = 1, 
                          double_rots = False
                          )
    data = next(iter(loader))
    rgb = data['img']
    flo = data['flo']
    rots_1 = data['rots_1']
    stream = UnifiedStream()
    
    x = stream(rgb,flo, rots_1 = rots_1)
    print(x.shape)
    
    # for _ in tqdm(loader):
    #     stopExec()