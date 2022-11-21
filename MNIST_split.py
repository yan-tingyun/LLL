from __future__ import print_function, division
from typing import Any, Dict, List, Tuple

import torch
from torchvision import datasets

from PIL import Image
import os.path



# class MNIST_Split(datasets.MNIST):
#     """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.
#     Args:
#         root (string): Root directory of dataset where ``processed/training.pt``
#             and  ``processed/test.pt`` exist.
#         train (bool, optional): If True, creates dataset from ``training.pt``,
#             otherwise from ``test.pt``.
#         download (bool, optional): If true, downloads the dataset from the internet and
#             puts it in root directory. If dataset is already downloaded, it is not
#             downloaded again.
#         transform (callable, optional): A function/transform that  takes in an PIL image
#             and returns a transformed version. E.g, ``transforms.RandomCrop``
#         target_transform (callable, optional): A function/transform that takes in the
#             target and transforms it.
#     """
  
    

#     def __init__(self, root, train=True, transform=None, target_transform=None, download=False,digits=[1,2]):
#         super(MNIST_Split, self).__init__(root, train, transform, target_transform, download)

#         #get only the two digits
#         self.digit_labels=None
#         self.digit_data=None
#         self.timestamp=None
#         self.classes= digits 
#         if self.train:
            
#             #loop over the given digits and extract there corresponding data
#             for digit in digits:
#                 digit_mask=torch.eq(self.targets , digit) 
#                 digit_index=torch.nonzero(digit_mask)
#                 digit_index=digit_index.view(-1)
#                 this_digit_data=self.data[digit_index]
#                 this_digit_labels=self.targets[digit_mask]
#                 # this_digit_labels.fill_(digits.index(digit))
#                 if self.digit_data is None:
#                     self.digit_data=this_digit_data.clone()
#                     self.digit_labels=this_digit_labels.clone()
#                     # self.timestamp=this_digit_labels.clone()
#                     # for pos in range(len(self.digit_labels)):
#                     #     # timestamp
#                     #     if pos < (len(self.digit_labels)*2/3) :
#                     #         if pos < (len(self.digit_labels)/3):
#                     #             self.timestamp[pos] = 2*digit
#                     #         else:
#                     #             self.timestamp[pos] = 2*digit+1

#                     #         self.digit_labels[pos] = 1
#                     #     else:
#                     #         if pos < (len(self.digit_labels)*5/6):
#                     #             self.timestamp[pos] = 2*digit
#                     #         else:
#                     #             self.timestamp[pos] = 2*digit+1

#                     #         self.digit_labels[pos] = 0
#                 else:
#                     self.digit_data=torch.cat((self.digit_data,this_digit_data),0)
#                     self.digit_labels=torch.cat((self.digit_labels,this_digit_labels),0)
                        
#             #self.data, self.targets = torch.load(
#                 #os.path.join(root, self.processed_folder, self.training_file))
#         else:
#                        #loop over the given digits and extract there corresponding data
#             for digit in digits:
#                 digit_mask=torch.eq(self.targets , digit) 
#                 digit_index=torch.nonzero(digit_mask)
#                 digit_index=digit_index.view(-1)
#                 this_digit_data=self.data[digit_index]
#                 this_digit_labels=self.targets[digit_mask]
#                 # this_digit_labels.fill_(digits.index(digit))
#                 if self.digit_data is None:
#                     self.digit_data=this_digit_data.clone()
#                     self.digit_labels=this_digit_labels.clone()

#                     # self.timestamp=this_digit_labels.clone()
#                     # for pos in range(len(self.digit_labels)):
#                     #     # timestamp
#                     #     if pos < (len(self.digit_labels)/2) :
#                     #         self.timestamp[pos] = 2*digit
#                     #     else:
#                     #         self.timestamp[pos] = 2*digit+1

#                     #     self.digit_labels[pos] = 0
                    
#                 else:
#                     self.digit_data=torch.cat((self.digit_data,this_digit_data),0)
#                     self.digit_labels=torch.cat((self.digit_labels,this_digit_labels),0)
                    
         
        
    
#     def __getitem__(self, index) -> Tuple[Any, Any]:
#         """
#         Args:
#             index (int): Index
#         Returns:
#             tuple: (image, target) where target is index of the target class.
#         """
       
#         img, target = self.digit_data[index], self.digit_labels[index]
       

#         # doing this so that it is consistent with all other datasets
#         # to return a PIL Image
#         img = Image.fromarray(img.numpy(), mode='L')

#         if self.transform is not None:
#             img = self.transform(img)

#         if self.target_transform is not None:
#             target = self.target_transform(target)
        
#         img=img.view(28*28,-1) 
#         # timestamp = torch.full((1,1), self.timestamp[index])
#         # img=torch.cat((img,timestamp), 0)
       
#         return img, target

#     def __len__(self):
#         return(self.digit_labels.size(0)) 



class MNIST_Split(datasets.MNIST):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
  
    

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False,digits=[1,2]):
        super(MNIST_Split, self).__init__(root, train, transform, target_transform, download)



        #get only the two digits
        self.digit_labels=None
        self.digit_data=None
        self.classes= digits 
        if self.train:
            
            #loop over the given digits and extract there corresponding data
            for digit in digits:
                digit_mask=torch.eq(self.targets , digit) 
                digit_index=torch.nonzero(digit_mask)
                digit_index=digit_index.view(-1)
                this_digit_data=self.data[digit_index]
                this_digit_labels=self.targets[digit_mask]
                this_digit_labels.fill_(digits.index(digit))
                if self.digit_data is None:
                    self.digit_data=this_digit_data.clone()
                    self.digit_labels=this_digit_labels.clone()
                else:
                    self.digit_data=torch.cat((self.digit_data,this_digit_data),0)
                    self.digit_labels=torch.cat((self.digit_labels,this_digit_labels),0)
            #self.data, self.targets = torch.load(
                #os.path.join(root, self.processed_folder, self.training_file))
        else:
                       #loop over the given digits and extract there corresponding data
            for digit in digits:
                digit_mask=torch.eq(self.test_labels , digit) 
                digit_index=torch.nonzero(digit_mask)
                digit_index=digit_index.view(-1)
                this_digit_data=self.test_data[digit_index]
                this_digit_labels=self.test_labels[digit_mask]
                this_digit_labels.fill_(digits.index(digit))
                if self.digit_data is None:
                    self.digit_data=this_digit_data.clone()
                    self.digit_labels=this_digit_labels.clone()
                    
                else:
                    self.digit_data=torch.cat((self.digit_data,this_digit_data),0)
                    self.digit_labels=torch.cat((self.digit_labels,this_digit_labels),0)
                    
         
        
    
    def __getitem__(self, index) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
       
        img, target = self.digit_data[index], self.digit_labels[index]
       

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        img=img.view(-1,28*28) 
       
        return img, target

    def __len__(self):
        return(self.digit_labels.size(0))