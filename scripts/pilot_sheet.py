import json,sys,os
from PIL import Image,ImageDraw,ImageFont
Fo=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",18)
def sheet(items,cols,path,W=272,H=352):
  R=(len(items)+cols-1)//cols; S=Image.new("RGB",(W*cols,(H+46)*R),"white"); d=ImageDraw.Draw(S)
  for i,(f,l1,l2) in enumerate(items):
    x,y=(i%cols)*W,(i//cols)*(H+46); im=Image.open(f).convert("RGB"); im.thumbnail((W,H)); S.paste(im,(x,y))
    d.text((x+4,y+H+2),l1,fill="black",font=Fo); d.text((x+4,y+H+23),l2,fill="black",font=Fo)
  S.save(path,quality=88)
