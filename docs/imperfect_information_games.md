# Established imperfect-information games

Partial observability matters because a strategic policy acts from an observation or information state while payoff-relevant world state remains hidden. Perfect-recall information states may encode private information and action history that an observation string omits. Conflating the two can create false claims about memory or opponent inference.

The OpenSpiel utilities support Kuhn Poker and Leduc Poker, legal-action enumeration, CFR+ reference-policy generation, NashConv, and exact continuation values under a reference policy. Reach-conditioned (Q(I,a)) is computed by integrating over underlying world states compatible with information state (I), forcing action (a), and following the reference policy thereafter.

In the frozen Leduc audit, exhaustive focal-player enumeration yielded 468 information states and 288 observations; 180 observation groups contained multiple information states. A CFR+ average policy after 5,000 iterations reached NashConv approximately (3.677\times10^{-5}). The abundant observation aliasing nevertheless produced negligible decision value under the frozen reference policy—an example of why structural partial observability alone does not prove strategically useful hidden information.

Run the small CPU demo with the optional OpenSpiel dependency:

```bash
pip install -e ".[openspiel]"
python3 scripts/run_openspiel_demo.py
```

The demo uses Kuhn for speed. It is an executable integrity check, not reproduction of the 5,000-iteration Leduc audit.
