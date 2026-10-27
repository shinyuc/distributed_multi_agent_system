# Pilot inputs: 16 scenarios of the 48-scenario feasibility study

Inputs only. Nothing in this directory starts the pilot. Run outputs belong under `outputs/` (git-ignored), never here.

The files were copied byte for byte from the pilot bundle that `python -m diagnostics.study48.bundle` built from the generated
study at commit 884560d. Two files differ from the bundle:

* `SERVER_STEPS.sh`: its paths were changed for in-repository use;
* `SHA256SUMS`: the checksum lines for `SERVER_STEPS.sh` and for this README.

The other 22 files are identical to the bundle and keep their original checksums.

| Path | Content |
|---|---|
| `configs/<ID>.yaml` | the 16 pilot configurations. SHA-256 values are in `manifest_pilot16.json`. |
| `warm/D03_warm.pkl` | frozen warm start of D03: stop step 863, E = 5.586e-3, bit-identical to the validated D3 state |
| `manifest_pilot16.json` | the 16 pilot entries of the study manifest, verbatim |
| `expected_outcomes.json` | Tier-0 predictions at x_0, for comparison after the run |
| `pilot/pilot_batch.yaml` | the pilot batch spec: 16 scenarios, `diagnostics.study48.run_pilot`, paths relative to this file |
| `pilot/server_smoke_batch.yaml` | smoke spec: C09 (zero obstacles), A01 (stops at step 0), D03 (warm start), 3 steps each |
| `SERVER_STEPS.sh` | prepares the server; never starts the pilot: <br>1. updates the research branch (fast-forward only) and checks the commit;<br>2. checks the checksums;<br>3. runs the tests;<br>4. runs and verifies the smoke batch;<br>5. prints the pilot command. |
| `BUNDLE_INFO.json` | bundle provenance. `source_study` is the generating machine's path; no code reads it. |
| `SHA256SUMS` | `sha256sum -c` format, all files above |

From the repository root (Linux):

```
EXPECT=<commit> bash experiments/feasibility_study/pilot16/SERVER_STEPS.sh          # preparation and smoke only
python -m diagnostics.run_batch --spec experiments/feasibility_study/pilot16/pilot/pilot_batch.yaml \
    --workers 16 --timeout 7200 --out outputs/feasibility_study/pilot16/pilot16_<timestamp>   # the pilot (separately)
```

Eight pilots are expected to stop at step 0 without applying a control: A01, A03, A04, A09, B01, B02, B04 and C03. These
are designed outcomes and are not to be modified. Feasibility labels:

* F, I and U are numerically verified verdicts;
* the certificates are sufficient conditions only;
* `upper_bound_below_tol` (0 <= rho <= upper bound <= 1e-9) means rho is numerically zero. It does not prove rho = 0.
