import argparse,json
from pathlib import Path
import esbgk as s
p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);a=p.parse_args()
root=Path(__file__).resolve().parent
g=json.loads((root/'PASS.json').read_text())
assert g['PASS'] and g['source']==s.ref.sha(root/'esbgk.py') and g['reference']==s.ref.sha(root/'shakhov_reference.py') and g['manifest']==s.ref.sha(root/'manifest.json')
c=json.loads((root/'manifest.json').read_text())['cases'][a.index]
assert not (root/'runs'/c['name']).exists(), 'Exclusive fresh case output required'
s.run(c)
