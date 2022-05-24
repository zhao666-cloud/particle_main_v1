from torch.utils.data import Dataset

class reconDataset(Dataset):
    def __init__(self,m_r_list,m_t_list,rgb_list,mask_list,intrinsic):
        self.m_r_list = m_r_list
        self.m_t_list = m_t_list
        self.rgb_list = rgb_list
        self.mask_list = mask_list
        self.intrinsic = intrinsic
    def __getitem__(self,index):
        sample = {
            'm_r_list' : self.m_r_list[index],
            'm_t_list' : self.m_t_list[index],
            'rgb_list' : self.rgb_list[index],
            'mask_list' : self.mask_list[index],
            'intrinsic' : self.intrinsic

        }
        return sample

    def __len__(self):
        return len(self.m_t_list)
