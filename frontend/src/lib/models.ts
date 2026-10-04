/** Fixed series order and encodings: colour + dash so identity never relies on colour alone. */
export const MODELS = [
  { name: "XGBoost", key: "xgboost", color: "var(--s-xgb)", hex: "#0d9488", dash: undefined },
  { name: "GraphSAGE", key: "graphsage", color: "var(--s-sage)", hex: "#6366f1", dash: "6 3" },
  { name: "HGT", key: "hgt", color: "var(--s-hgt)", hex: "#d97706", dash: "2 3" },
  { name: "HeteroRGCN", key: "rgcn", color: "var(--s-rgcn)", hex: "#fb7185", dash: "8 3 2 3" },
] as const;
