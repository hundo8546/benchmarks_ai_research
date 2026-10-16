# Exact pseudocode for the binary gate and the three-way selector, as implemented in `routing/gate_analysis.py`

This file documents the logic exactly as it exists in `routing/gate_analysis.py`
(`oracle_decisions()`, lines 25-34, and the bootstrap loop inside `sweep_cell()`,
lines 37-75). It is read-only documentation; no existing file was modified.

## 1. Oracle decision (full disagreement subset used as calibration)

```text
function oracle_decisions(cell):
    y_true, y1, y2, y3 = cell.y_true, cell.y_cheap1, cell.y_cheap2, cell.y_verifier
    acc1 = mean(y1 == y_true)      # f1 (artifact detector) accuracy on full subset
    acc2 = mean(y2 == y_true)      # f2 (semantic detector) accuracy on full subset
    acc3 = mean(y3 == y_true)      # f3 (verifier) accuracy on full subset
    best_cheap_acc = max(acc1, acc2)

    binary_escalate   = (acc3 > best_cheap_acc)              # STRICT inequality; Boolean ONLY
    three_way_choice  = argmax([acc1, acc2, acc3])            # 0=f1, 1=f2, 2=f3
    return acc1, acc2, acc3, binary_escalate, three_way_choice
```

Tie handling: `np.argmax` returns the **first** index attaining the max. For
the list order `[acc1, acc2, acc3]` this means: on an exact tie between f1
and f2, `three_way_choice` picks **f1** (index 0); on an exact tie between
`max(f1,f2)` and f3, `three_way_choice` picks whichever cheap detector is
tied (index 0 or 1), i.e. **ties always favor a cheap detector over the
verifier**, and among cheap detectors **always favor f1 over f2**. This was
confirmed directly in code (`int(np.argmax([acc1, acc2, acc3]))`) and
empirically in the divergence analysis below (Section 3): at N=10, exact
f1/f2 ties are resolved to `cheap1` in 100% of observed tied trials.

The binary rule's `binary_escalate` uses a **strict `>`**, so on an exact
tie `acc3 == best_cheap_acc`, `binary_escalate = False` (no escalation) —
consistent in spirit with "ties go to the cheap side" that the three-way
rule's tie-break also implements.

## 2. Bootstrap-calibrated decision, per resample (`sweep_cell`)

```text
function sweep_cell(cell, rng):
    acc1, acc2, acc3, oracle_binary, oracle_three_way = oracle_decisions(cell)
    for N in [10, 20, 50, 100, 200, "oracle"]:
        binary_matches = 0
        three_way_matches = 0
        repeat N_RESAMPLES=1000 times:
            idx = rng.choice(n_disagree, size=N, replace=True)   # same idx used for all of a1,a2,a3
            a1 = mean(y1[idx] == y_true[idx])
            a2 = mean(y2[idx] == y_true[idx])
            a3 = mean(y3[idx] == y_true[idx])

            calib_binary    = (a3 > max(a1, a2))                 # Boolean
            calib_three_way = argmax([a1, a2, a3])               # 0/1/2, same tie rule as above

            binary_matches    += (calib_binary    == oracle_binary)     # <-- compares BOOLEANS only
            three_way_matches += (calib_three_way == oracle_three_way)  # <-- compares the full 3-way identity

        record binary_agreement_rate    = binary_matches    / N_RESAMPLES
        record three_way_agreement_rate = three_way_matches / N_RESAMPLES
```

## 3. The crux finding, stated precisely

`gate_analysis.py` **never computes, stores, or compares which of f1/f2 the
binary rule "falls back to."** There is no variable anywhere in the file
representing "the binary rule's chosen cheap detector." The only quantity
tracked for the binary rule is the single Boolean `binary_escalate` /
`calib_binary` (escalate vs. don't-escalate), and `binary_agreement_rate` is
the agreement rate of **that Boolean alone** against the oracle Boolean.

By contrast, `three_way_agreement_rate` requires the full 3-valued
`calib_three_way` label (`cheap1` / `cheap2` / `verifier`) to match the
oracle's 3-valued label exactly.

**This means the two reported "agreement rate" columns in the paper's
Table `tab:sensitivity` are not measuring agreement with the same granularity
of decision.** The binary column measures agreement on a coarser 2-outcome
quantity (escalate / don't-escalate). The three-way column measures
agreement on a finer 3-outcome quantity (which of f1, f2, f3 is used). Any
trial in which the calibration sample incorrectly swaps f1 for f2 (or vice
versa) while correctly keeping the system in the "don't escalate" branch is
counted as a **match** for the binary rule and a **mismatch** for the
three-way rule. This is precisely what produces the N=10 divergence (Section
3 below) and it is a difference in *what was measured* in `gate_analysis.py`,
not evidence that the underlying rules disagree about *which action to take*
in that regime (see Section 4).

## 4. Are the two rules equivalent as *decisions* (actions), or only on the escalate-vs-not Boolean?

**As stated in the paper's prose** ("Enable the cascade only if
$A_{V,cal} > \max(A_{1,cal}, A_{2,cal})$... Otherwise, fall back to the
stronger cheap detector"), the binary gate IS a complete three-way decision
procedure: it names an action for all three cases (escalate to f3, or run
whichever of f1/f2 is "stronger"). Comparing this prose rule to the
three-way selector's `argmax([acc1, acc2, acc3])`:

- If f3 wins (`acc3 > max(acc1, acc2)`): both rules escalate to f3. Identical.
- If f3 does not win: the binary gate's prose says "fall back to the
  stronger cheap detector," i.e. `argmax([acc1, acc2])` restricted to the
  cheap pair — which is **exactly** the cheap-side case of the three-way
  rule's `argmax([acc1, acc2, acc3])` whenever f3 isn't the max. So the two
  rules select the **same detector** in every case, modulo the unspecified
  tie-break in the prose (the code implements no such tie-break for the
  binary rule at all, because it never computes fallback identity — see
  Section 3).

**Conclusion: the two rules are mathematically equivalent as decision rules
(same action in every case) — the reviewer is correct.** They are *not*
equivalent as implemented *analyses* in `gate_analysis.py`, because the
code's binary-rule agreement metric only ever tests the coarser
escalate/don't-escalate Boolean and never tests fallback-detector identity.
That implementation gap, not a conceptual difference in the rules
themselves, is what the paper needs to correct (see `RESPONSE_DRAFT.md`).
