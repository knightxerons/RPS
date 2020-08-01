import os
from PIL import Image

path = os.listdir('./')

for i in path:
    if i.endswith('.jpg'):     
        print(i)
        img = Image.open(i) 
        img = img.resize((64,64), Image.ANTIALIAS)
        img.save(i)
