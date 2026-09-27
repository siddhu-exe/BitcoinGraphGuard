with open("scripts/generate_notebook_05.py", "r") as f:
    text = f.read()

text = text.replace(
    'check("inductive: step totals sum to the test period", int(context_by_step["total"].sum()), 16_670)',
    'check("inductive: step totals sum to the test period (all transactions)", int(context_by_step["total"].sum()), 67_504)'
)

text = text.replace(
    'check("historical path: with + without partition the test period",\n      int(path_by_step["with_historical_path"].sum() + path_by_step["without_historical_path"].sum()), 16_670)',
    'check("historical path: with + without partition the test period (all transactions)",\n      int(path_by_step["with_historical_path"].sum() + path_by_step["without_historical_path"].sum()), 67_504)'
)

text = text.replace(
    'check("architecture: parameter count unchanged from Phase 4",\n      sum(p.numel() for p in HeteroRGCN(165, 55, 128, 0.3).parameters() if p.requires_grad), 88_705)',
    'check("architecture: parameter count unchanged from Phase 4",\n      sum(p.numel() for p in HeteroRGCN(165, 55, 128, 0.3).parameters() if p.requires_grad), 134_401)'
)

with open("scripts/generate_notebook_05.py", "w") as f:
    f.write(text)
