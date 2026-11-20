"""Validate a downloaded official PVT checkpoint and record its source receipt."""
import argparse
from common import ROOT, torch, sha256, atomic_json
from implementation.model import Field

SOURCES = {
    'https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth':
        'PVT classification README, PVTv2 on ImageNet-1K, PVT-V2-B2 row: https://github.com/whai362/PVT/blob/v2/classification/README.md',
    'https://drive.google.com/file/d/1uzeVfA4gEQ772vzLntnkqvWePSw84F6y/view?usp=sharing':
        'Official HBGNet README ImageNet pretrained PVT-v2-B2 link: https://github.com/NanNanmei/HBGNet',
    'https://drive.google.com/file/d/1snw4TYUCD5z4d3aaId1iBdw-yUKjRmPC/view?usp=sharing':
        'PVT classification README PVT-V2-B2 alternate Google Drive link',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-url',required=True,choices=sorted(SOURCES))
    args = parser.parse_args()
    path = ROOT/'pretrained/pvt_v2_b2.pth'
    receipt = path.with_suffix('.json')
    if receipt.exists(): raise FileExistsError('Do not overwrite an existing provenance receipt')
    digest = sha256(path)
    torch.set_num_threads(2)
    model = Field(pretrained_path=str(path))
    if sha256(path) != digest: raise ValueError('Checkpoint changed during verification')
    atomic_json(receipt,{'sha256':digest,'bytes':path.stat().st_size,'source_url':args.source_url,
        'source_evidence':SOURCES[args.source_url], 'backbone_keys_loaded':len(model.backbone.state_dict()),
        'validation':'weights_only CPU load; all backbone keys; strict shapes; finite tensors',
        'integrity_limit':'Source URL is supplied by the operator. No publisher-provided checksum was available; local SHA256 pins these bytes, not publisher identity.'})
    print('Validated backbone and wrote',receipt)


if __name__=='__main__': main()
