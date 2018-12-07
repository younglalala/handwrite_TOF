import os
from PIL import Image





img_path='/Users/wywy/Desktop/level_choice/all_level'
save_path='/Users/wywy/Desktop/level_choice/all_aug'
c=0
for file in os.listdir(img_path):
    if file=='.DS_Store':
        os.remove(img_path+'/'+file)
    else:
        img=Image.open(img_path+'/'+file)
        img.save(save_path+'/'+file)
        c+=1
        print(save_path+'/'+file,'------',c)

