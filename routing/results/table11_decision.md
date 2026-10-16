# Table 11 (tab:transfer) number decision

The old 0.917 figure (GenBuster-trained learned-escalation-classifier transfer accuracy)
is not reproducible from any current on-disk data (see the confidence_routing agent's
report: current standard protocol gives byte-identical predictions, classifier=disagreement=0.899,
Delta=0.000; a legacy internal-split protocol reproduced on today's data gives 0.900/0.903,
Delta=-0.0033, CI=[-0.0100, 0.0000]).

DECISION: use the current standard-protocol number (0.899 vs 0.899) as the primary
Table 11 entry, since it is what the current, single, consistent, reproducible pipeline
produces end-to-end. Cite the legacy-protocol reproduction (0.900/0.903) as a footnote/
robustness check showing the "no measurable improvement" conclusion holds under either
protocol -- the two numbers disagree slightly but agree on the conclusion that matters.
