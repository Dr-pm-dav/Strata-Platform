# STRATA — Methods & Publication Notes

This note documents the methodology behind STRATA and maps the platform to
concrete, publishable contributions. It is written so sections can be lifted
into manuscripts with Dr. Qu (corresponding author, yqu@coloradotech.edu).

> Scope note: all results produced offline use the synthetic reflectance
> substrate and are **methodological demonstrations**, not empirical findings.
> Empirical results require the workstation connectors (HLS, USGS, NREL) and
> the real auxiliary layers named below. Every quantitative claim intended for
> publication must be regenerated on real data.

---

## 1. The SHAD-RD engine, generalized

SHAD-RD (the dissertation framework) is, abstractly, a four-part pipeline:

1. **Reflectance** from a harmonized satellite source (HLS v2.0).
2. **Features** = physical spectral indices (+ optional Prithvi-EO-2.0-300M
   embeddings).
3. **Model** = LightGBM (regression or classification).
4. **Validation** = spatially-blocked cross-validation (SBCV), reported
   alongside random K-fold to expose spatial information leakage.

The dissertation instantiated this for water-quality turbidity regression and
reported the signature SBCV result: a model that looks skilful under a random
split degrades sharply under spatial blocking (random-split R² positive,
HUC-12 SBCV R² negative). STRATA holds parts 1–4 fixed and changes only the
target, demonstrating that the *framework*, not the water-quality application,
is the contribution.

---

## 2. Spatial-leakage discipline in mineral prospectivity (primary new result)

Mineral prospectivity models are trained on spatially clustered presence
points (known deposits) and pseudo-absences. This is exactly the setting where
random cross-validation leaks: test points fall near training points from the
same cluster, and the reported skill does not transfer to genuinely new
ground.

STRATA's `minerals.prospectivity` applies the dissertation's SBCV discipline to
this problem. `shadrd.leakage_report` fits the model under both schemes and
reports the **leakage gap** (random minus spatial). On the synthetic substrate
— in which the alteration signal is deliberately **non-stationary** (different
deposits express alteration in different bands, controlled by the `signal`
knob, with `regional` adding a smooth confounder) — the framework recovers a
positive leakage gap, and setting `signal=0` collapses both schemes to chance,
confirming the detector behaves as designed.

**Publishable claim:** prospectivity AUC under random CV systematically
overstates transferable skill; SBCV is the honest estimator; the gap is
quantifiable and should be reported as standard practice. This directly extends
**Article 1 ("Spatial Leakage in HLS Turbidity Regression: A Dual-Basin SBCV
Benchmark")** from water to mineral targeting — a second domain for the same
methodological point strengthens the benchmark paper or supports a companion.

Real-data requirement: HLS-derived alteration indices + USGS geophysics
(aeromagnetics, radiometrics) + Earth MRI geochemistry as features; MRDS
presence points; HUC-12 or geologic-terrane blocking.

---

## 3. Cross-sector transferability (platform-level result)

A single `ShadrdModel` and a single feature contract serve water-quality
regression, mineral-prospectivity classification, and the geospatial scoring
that feeds siting. The shared engine is the evidence for transferability.

**Publishable claim:** a geospatial-foundation-model + gradient-boosting +
SBCV pipeline transfers across environmental-monitoring tasks with no change to
the learning machinery — only the target and auxiliary layers change. This is a
natural new manuscript ("A transferable remote-sensing ML framework for
strategic-resource monitoring: from water quality to critical minerals"),
complementary to **Article 2 ("SHAD-RD: Open-Source Dual-Basin Water Analytics
on Commodity Hardware")**; STRATA is the open-source artifact that paper can
cite as the reference implementation.

---

## 4. Supply-demand gap → site linkage (decision-support result)

`supply.gap` projects demand (clean-energy-driven CAGR, scenario-scaled) against
a domestic-supply ramp, ranks commodities by a composite shortfall score
(terminal gap × import reliance × criticality weight), and — the integrative
step — resolves the top commodities to specific catalog sites via
`supply.priority_sites`. The macro shortfall analysis therefore terminates in a
map, not a table.

**Publishable angle:** a reproducible, open pipeline linking national
critical-mineral shortfall projections to site-level monitoring priorities;
methodologically modest but useful as an applied/policy contribution, and a
clean fit for a workshop or applied-track venue. Real-data requirement: current
USGS Mineral Commodity Summaries; published demand scenarios (IEA / DOE).

---

## 5. Clean-firm-power siting (applied geospatial result)

`energy.siting` scores candidate cells on six transparent, separately-weighted
criteria, including a **minerals-link** term that rewards co-locating clean firm
generation with the critical-mineral demand it would electrify (reducing new
transmission). Scoring is intentionally multi-criteria rather than learned:
there are no siting labels, so an auditable weighted score is the honest choice.

**Publishable angle:** an open multi-criteria siting tool coupling energy
build-out to the resource base; suited to an energy-systems or
applied-geospatial venue. This complements the energy-systems direction of the
broader portfolio (the Helix line of work). Real-data requirement: USGS 3DEP
(slope), NHD/NWIS (water), HIFLD (transmission), PAD-US (exclusions), and a
load/transmission model.

---

## Suggested sequencing with Dr. Qu

1. Fold the **mineral-prospectivity SBCV** result into the Article 1 benchmark
   (or a tight companion) — lowest marginal effort, highest methodological
   payoff, and it reuses the dissertation's central finding.
2. Write the **cross-sector transferability** paper around STRATA as the
   reference implementation (pairs with Article 2).
3. Treat the **gap→site** and **siting** pieces as applied/policy short papers
   or talks once real layers are wired.

All four map cleanly onto your existing EJECE/IEEE plan; STRATA gives every one
of them a runnable, inspectable artifact reviewers can execute.
