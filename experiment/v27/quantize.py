"""Quantize the constant transformer projections (the upstream graph quantizes its embedding table)."""
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile,ZIP_DEFLATED
from .data import ROOT,OUT,sha

def main():
    from onnxruntime.quantization import QuantType,quantize_dynamic
    root=ROOT/'app/weights'
    parent=root/'intent-review'
    target=ROOT/'artifacts/v27-quantized'
    target.mkdir(exist_ok=True)
    manifest=json.loads((parent/'encoder.json').read_bytes())
    with TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
        original=Path(tmp)/'original.onnx'
        with original.open('wb') as dest:
            for name in manifest['parts']:
                with ZipFile(parent/name) as z:dest.write(z.read('chunk'))
        assert sha(original)==manifest['graph_sha256']
        quant=Path(tmp)/'recipient.int8.onnx'
        quantize_dynamic(str(original),str(quant),weight_type=QuantType.QInt8,
                         op_types_to_quantize=['MatMul','Gemm'],per_channel=True,
                         extra_options={'MatMulConstBOnly':True})
        parts=[]
        with quant.open('rb') as source:
            while chunk:=source.read(52*1024*1024):
                name=f'int8-{len(parts)}.zip'
                with ZipFile(target/name,'w',compression=ZIP_DEFLATED,compresslevel=3) as z:
                    z.writestr('chunk',chunk)
                parts.append(name)
        (target/'int8.json').write_text(json.dumps(dict(parts=parts,graph_sha256=sha(quant),
            parent=manifest['graph_sha256']),indent=2),encoding='utf8')
        print(quant.stat().st_size,len(parts),flush=True)

if __name__=='__main__':main()
