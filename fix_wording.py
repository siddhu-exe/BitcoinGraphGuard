import re
with open("docs/TEMPORAL_INDUCTIVE_EVALUATION.md", "r") as f:
    text = f.read()

text = re.sub(
    r'This confirms our Phase 4 hypothesis that uniform, unweighted mean-aggregation across the dense `AddrAddr` graph causes extreme message diffusion and over-smoothing\.',
    'The results support the hypothesis that uniform relational aggregation may fail to distinguish useful historical address context from noisy recurring connectivity.',
    text
)
with open("docs/TEMPORAL_INDUCTIVE_EVALUATION.md", "w") as f:
    f.write(text)
