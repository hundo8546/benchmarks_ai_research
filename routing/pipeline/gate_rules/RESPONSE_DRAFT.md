# Draft response to Reviewer 1, Point 3 (validation gate vs. three-way selector)

*Grounded in `routing/gate_analysis.py` (`oracle_decisions`, `sweep_cell`) and
`routing/results/gate_sensitivity_sweep.csv`. See `PSEUDOCODE.md` and
`explain_n10_divergence.py` in this directory for the supporting code and
numbers.*

**Paragraph 1 (acknowledge the equivalence).** The reviewer is correct, and
we thank them for pressing on this point. As stated in Section
`sec:validation_gate`, the binary gate escalates to the verifier $f_3$ when
$A_{3,cal} > \max(A_{1,cal}, A_{2,cal})$ and otherwise "falls back to the
stronger cheap detector," i.e. $\arg\max(A_{1,cal}, A_{2,cal})$. This is, by
construction, the restriction of the three-way selector's
$\arg\max(A_{1,cal}, A_{2,cal}, A_{3,cal})$ to the two cases "the verifier
wins" and "the verifier does not win." The two rules therefore select the
*same detector in every case* (modulo tie-breaking, which our
implementation resolves identically in spirit — ties favor the cheaper
option): the binary gate is not a different decision procedure from the
three-way selector, it is the same procedure described in two-step rather
than one-step form. We will revise Section `sec:threeway` to state this
equivalence explicitly rather than presenting the three-way rule as a
strictly more expressive generalization.

**Paragraph 2 (correct the "cannot represent" claim).** The claim in
Section `sec:threeway` that preferring the artifact detector $f_1$ "is the
failure mode the binary escalate-or-fallback rule cannot represent" is
incorrect and will be removed/corrected. Because the binary gate's fallback
is defined as "the stronger cheap detector," whenever $A_{1,cal} >
A_{2,cal}$ and the verifier does not win, the deployed binary-gate system
already runs $f_1$ — it represents exactly the same artifact-detector
preference the three-way selector does. The actual distinction between the
two formulations in our released analysis is narrower and purely about
*what we measured*, not what the rule can *do*: our calibration-sensitivity
code (`gate_analysis.py`) tracks, for the binary gate, only the Boolean
escalate/don't-escalate decision and compares that Boolean to the oracle's
Boolean; it never separately records or validates which of $f_1$/$f_2$ the
binary gate's fallback names. The three-way sensitivity sweep, by
construction, does track and validate the fallback detector's identity. We
will rewrite this subsection to state that the three-way formulation's
contribution is making the fallback detector's identity an explicit,
separately-calibrated and separately-validated quantity — valuable for
reporting (Table `tab:threeway`'s breakdown by detector) and for auditing
calibration reliability at the level of *which model runs*, not for
expanding the set of reachable actions.

**Paragraph 3 (explain the N=10 divergence mechanically).** The $N=10$ row
of Table `tab:sensitivity` (13/35 binary vs. 12/35 three-way) comes from a
single configuration: `limited_data / SD14+SigLIP2, frac=100% / verifier =
Qwen2.5-VL-7B` (n=532 disagreement examples). On the full subset,
$A_1=21.6\%$ (FFT), $A_2=78.4\%$ (SigLIP2), $A_3=58.7\%$ (Qwen2.5-VL-7B); the
oracle does not escalate and the oracle three-way choice is $f_2$. Because
this is a disagreement subset with binary labels, $A_1$ and $A_2$ are
exactly complementary ($A_1+A_2=1$) at every sample size. At $N=10$, the
resampled $a_1$ can only take the 11 values $\{0, 0.1, \dots, 1.0\}$, so an
exact tie $a_1=a_2=0.5$ occurs whenever precisely 5 of the 10 resampled
points are "$f_1$-correct" — probability $\binom{10}{5}A_1^5A_2^5\approx
3.5\%$ by the binomial PMF, matching the 35/1000 bootstrap resamples we
observe landing exactly on this tie. Our tie-break (`argmax` returns the
first index) resolves every such tie to $f_1$, even though $f_1$'s true
accuracy is only 21.6%. In 23 of those 35 tied resamples, the verifier still
does not exceed the tied value, so the binary gate's Boolean ("don't
escalate") still agrees with the oracle — the binary metric is indifferent
to the fact that its named fallback detector flipped to $f_1$ — while the
three-way metric requires the exact label $f_2$ and counts these 23
resamples as mismatches. $23/1000=2.3$ percentage points is exactly the gap
between the binary rate (91.6%, above the 90% cutoff) and the three-way rate
(89.3%, below it) for this one configuration, and it is this single
configuration crossing the cutoff in opposite directions for the two
metrics that changes the aggregate count from 13/35 to 12/35. At $N\geq20$
the resampled-accuracy grid is finer ($1/20, 1/50,\dots$), the exact-tie
probability collapses rapidly, and the two rules' per-configuration rates
move together above/below 90% at every configuration, reproducing the
identical 19/35, 21/35, 24/35, 28/35 totals reported at $N=20,50,100,200$.
We will add a short footnote explaining this mechanism in place of the
current sentence that only reports the bare difference ("...trails by one
configuration").
