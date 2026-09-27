import re
with open("docs/PLAN.md", "r") as f: text = f.read()

text = re.sub(r'\*Status: NOTEBOOK AUTHORED, AWAITING REMOTE RUN.*?\* \[x\] Author the Phase 5 notebook and its methodology document',
r'*Status: COMPLETE.* \n\n* [x] Author the Phase 5 notebook and its methodology document', text, flags=re.DOTALL)

text = text.replace('* [ ] Evaluate the frozen', '* [x] Evaluate the frozen')
text = text.replace('* [ ] Measure label-free', '* [x] Measure label-free')
text = text.replace('* [ ] Compare static', '* [x] Compare static')
text = text.replace('* [ ] Test one', '* [x] Test one')
text = text.replace('* [ ] Break RGCN', '* [x] Break RGCN')
text = text.replace('* [ ] Recompute historical', '* [x] Recompute historical')
text = text.replace('* [ ] Execute on', '* [x] Execute on')

with open("docs/PLAN.md", "w") as f: f.write(text)
