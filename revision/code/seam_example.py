from pathlib import Path
import json
import numpy as np,rasterio
from PIL import Image,ImageDraw,ImageFont
from eval_revision import BASE,atomic
def main():
    dirs={r:BASE/'jobs'/('seam_'+r) for r in ('hbg_73','rgv_73')}
    rows=[json.loads(s) for s in (dirs['rgv_73']/'per_parent.jsonl').read_text().splitlines()]
    assert len(rows)==862
    examples=sorted([r for r in rows if r['disagreement_on_seam']>0],key=lambda r:(r['disagreement_on_seam'],r['id']))
    chosen=examples[(len(examples)-1)//2] if examples else rows[0];pid=chosen['id'];root=Path('/root/wyk5/dataset/AI4Boundaries_1m')
    with rasterio.open(root/'images/test'/(pid+'_ortho_1m_512.tif')) as f:rgb=f.read([1,2,3]).transpose(1,2,0).astype('uint8')
    with rasterio.open(root/'masks/test'/(pid+'_ortholabel_1m_512.tif')) as f:truth=f.read(1)>.5
    panels={'Image':rgb,'Reference':truth}
    mm={}
    for name,r in (('HBGNet','hbg_73'),('RGVNet','rgv_73')):
        for mode in ('direct','overlap'):
            p=dirs[r]/'masks'/(pid+'_'+mode+'.bin');mm[(r,mode)]=np.unpackbits(np.fromfile(p,np.uint8)).reshape(512,512).astype(bool)
            panels[name+' '+mode]=mm[(r,mode)]
    y,x=np.ogrid[:512,:512];band=((y>=248)&(y<264))|((x>=248)&(x<264));coords=np.argwhere(band&(mm[('rgv_73','direct')]!=mm[('rgv_73','overlap')]))
    cy,cx=coords[len(coords)//2] if len(coords) else (256,256)
    x0=int(np.clip(cx-48,0,416));y0=int(np.clip(cy-48,0,416));box=(x0,y0,x0+96,y0+96)
    out=Image.new('RGB',(6*226,477),'white');draw=ImageDraw.Draw(out)
    for j,(name,a) in enumerate(panels.items()):
        im=Image.fromarray(a if a.ndim==3 else (a.astype('uint8')*255)).convert('RGB')
        crop=im.crop(box);v=ImageDraw.Draw(im);v.line((256,0,256,511),fill=(53,140,178));v.line((0,256,511,256),fill=(53,140,178));v.rectangle(box,outline=(230,137,10),width=3)
        out.paste(im.resize((220,220)),(j*226,25));out.paste(crop.resize((220,220),Image.Resampling.NEAREST),(j*226,251));draw.text((j*226+8,6),name,fill='black')
    out.save('seam_example.png');atomic(Path('complete.json'),{'parent':pid,'example_rule':'lower median number of changed seam pixels among nonzero RGVNet parents, ties by parent ID; fixed center at middle changed pixel in raster order','nonzero_parents':len(examples),'box':list(box),'changed':chosen['disagreement_on_seam']})
if __name__=='__main__':main()

