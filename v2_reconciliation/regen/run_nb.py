"""run_nb.py — execute the notebook cell-by-cell with nbclient (nbconvert is broken in this env),
printing real-time per-cell progress, saving outputs in place, and timing against the 10-min budget."""
import sys
import time

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

REPO = "/home/ray/default/robotics_world_model_inference/v2_reconciliation"
path = f"{REPO}/world_model_reconciliation.ipynb"

nb = nbformat.read(path, as_version=4)
client = NotebookClient(nb, timeout=600, kernel_name="python3")
t0 = time.time()
failed = None
with client.setup_kernel():
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        ts = time.time()
        try:
            client.execute_cell(cell, i)
            print(f"[cell {i:>2}] OK   {time.time()-ts:5.1f}s   (total {time.time()-t0:5.0f}s)", flush=True)
        except CellExecutionError as e:
            print(f"[cell {i:>2}] FAIL {time.time()-ts:5.1f}s", flush=True)
            print("----- source -----\n" + cell.source[:1500], flush=True)
            print("----- error -----\n" + str(e)[:2500], flush=True)
            failed = i
            break

nbformat.write(nb, path)
total = time.time() - t0
if failed is None:
    print(f"\nDONE — notebook executed CLEAN in {total:.0f}s ({total/60:.1f} min); budget 600s.", flush=True)
    sys.exit(0)
else:
    print(f"\nFAILED at cell {failed} after {total:.0f}s.", flush=True)
    sys.exit(2)
