# 文件作用：提供数据集读取类SubjectIndependentTestDataset
# 加载视频转换的YUV序列并按滑动窗口划分
# written by Yongzhi
import torch
import numpy as np
import os

class SubjectIndependentTestDataset(torch.utils.data.Dataset):
    """
    该数据集从指定的路径加载领域数据，
    并以长度128、步长1的滑动窗口分割
    """
 
    def __init__(self, load_path:str=r'content/data', name:str='demo'):
        """
        参数:
            load_path(str): 数据所在的路径
            name(str): The name of the dataset files (e.g., 'demo')
            window(int): 滑动窗口长度
            step(int): 滑动窗口步长
            img_size(int): 方形图像尺寸
        """

        self.load_path = load_path
        self.name = name
        self.window = 128
        self.step = 1
        self.img_size = 8
        self.frames = np.load(os.path.join(self.load_path,self.name+'.npy'))
        self.Frames = np.zeros((self.frames.shape[0],self.img_size,self.img_size,3),dtype=np.float32)    
        
        for i in range(0,self.frames.shape[0]):
            frame = np.array(self.frames[i,:,:,:].copy(),dtype=np.float32)/255.0            
            self.Frames[i,:,:,:] = frame
       
        # Load ground truth file
        self.y_file = self.getFullGTfile()        
        # Load timestamp file
        self.t_file = self.getFulltimeFile()
        # Get windows index
        self.windows = self.getWindows()
            
    # 返回窗口数量
    def __len__(self):
            return len(self.windows)

    # 返回第 i 个样本
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Take only current window from big tensor
        frames = self.take_frames_crt_window(self.windows[idx])
        GT = torch.tensor(self.y_file[self.windows[idx][0]:self.windows[idx][1]], dtype=torch.float32)
        time = torch.tensor(self.t_file[self.windows[idx][0]:self.windows[idx][1]], dtype=torch.float32)
        sample = {'x':frames, 'y':GT, 't':time}
        return sample    

    # 获取窗口索引的函数
    def getWindows(self):
        windows = []
        for i in range(0,np.size(self.Frames,0)-self.window+1,self.step):
            windows.append((i,i+self.window))
        return windows

    # 从当前窗口提取帧并返回张量
    def take_frames_crt_window(self,idx):
        frames = torch.zeros((3,self.window,self.img_size,self.img_size)) # list with all frames {3,T,128,128}
        
        # Load all frames in current window
        for j,i in enumerate(range(idx[0],idx[1])):
            frame = self.Frames[i,:,:,:]
            frame = torch.tensor(frame, dtype=torch.float32).permute(2,0,1)#In pythorch channels must be in position 0, cv2 has channels in 2
            frames[:,j,:,:] = frame 
        return frames

    # 获取完整GT文件的函数
    def getFullGTfile(self):
        return np.loadtxt(os.path.join(self.load_path,self.name+'_gt.txt'))

    # 获取完整时间戳文件的函数  
    def getFulltimeFile(self):
        return np.loadtxt(os.path.join(self.load_path,self.name+'_timestamp.txt'))
        
