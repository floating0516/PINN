import yaml, sys
sys.path.insert(0, "/Users/lihe/Library/Mobile Documents/com~apple~CloudDocs/PINN_Mag/demo")
from src.models.model import PINNModel
base = "/Users/lihe/Library/Mobile Documents/com~apple~CloudDocs/PINN_Mag/demo/outputs_experiments/e1_backbone/models/"
for name, d in [("hybrid","20260623_100405"),("tcn_only","20260623_102522"),("transformer_only","20260623_103803")]:
    cfg = yaml.safe_load(open(base+d+"/config.yaml"))
    m = PINNModel(cfg)
    n = sum(p.numel() for p in m.parameters())
    mc = cfg.get("model",{})
    print(f"{name:18s} tcn={mc.get('num_tcn_blocks')} trans={mc.get('transformer_num_layers')} params={n} ({n/1e6:.3f}M)")
