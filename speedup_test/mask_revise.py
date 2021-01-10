import torch
mmask=torch.load('mask.pth')
mask_new={}
inf_mask={}
for key in mmask:
    print(key)
    #mask_new[key[7:]]=mmask[key]
    #print(mask_new)
    inf_mask[key[7:]]=mmask[key]
    #if 'bn3' in key:continue
    mask_new[key[7:]]=mmask[key]
    if 'bn' in key and 'bn3' not in key:
        mask_new[key[7:]]['weight']=mask_new[key[7:]]['bias']
print(mask_new.keys())
print(inf_mask)
print('=='*20)
print(mask_new)
torch.save(inf_mask,'mask_old.pth')
torch.save(mask_new,'mask_new.pth')
