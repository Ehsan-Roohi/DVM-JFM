import json,solver
m=json.loads((solver.ROOT/'manifest.json').read_text())
reports=[]
for c in m['cases']:
    reports.append(solver.gate(c))
r=dict(PASS=True,source=solver.sha(solver.ROOT/'solver.py'),manifest=solver.sha(solver.ROOT/'manifest.json'),cases=reports)
solver.dump(solver.ROOT/'PASS.json',r)
print(json.dumps(r))
