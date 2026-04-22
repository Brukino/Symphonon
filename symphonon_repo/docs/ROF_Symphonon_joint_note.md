# ROF-Symphonon Interface — Joint Formal Note

*Dario Panerati & Gareth Wilson — working draft, 2026-04-17*
*Not for circulation — in preparation*

---

## Premise

This note formalises the structural intersection between the Referential Ontology
Framework (ROF, Wilson) and Symphonon (Panerati). It does not claim unification.
It identifies three concrete mappings that are formally precise and empirically
testable, and one open question that neither framework can currently answer alone.

---

## 1. Core mapping

ROF defines a referent as R{A, r, B} where:
- R is the referential space (the scene within which A and B can meaningfully relate)
- r is the realised relationship at instance t
- r(t) is the total of what has happened, not what can possibly happen

Symphonon defines a referent (Gestalt G) as a cluster satisfying:

```
∂(G)(t) = |⟨e^{iφ}⟩_internal − ⟨e^{iφ}⟩_external| > θ_boundary
```

sustained by an energy balance:

```
dV/dt = gain(T_local) − decay > 0
```

**The mapping:**

```
R  (referential space)        ↔  field configuration (φ, ψ) at time t
   (membrane / identity)      ↔  ∂(G) — boundary condition, measurable
   (volume / possibility)     ↔  distribution of T_local across field
r(t) (realised relationship)  ↔  T_local(t) — instantaneous field density
                                  at referent position
```

r(t) in ROF is not a static evaluation — it is a time series. Symphonon computes
it at every simulation step, for every referent, in every field zone.

---

## 2. Enclosure condition and evaluability

ROF states: if Sign and Object are totally enclosed within R, the relationship
between them can always be truly evaluated. If not, the relationship is not
always True — potential paradox.

In Symphonon, this maps onto a measurable field condition:

```
Enclosed:     ∂(G) >> θ_boundary  AND  T_local >> T*
              → Sign-Object relationship stable, evaluable

Marginal:     ∂(G) ≈ θ_boundary  OR   T_local ≈ T*
              → relationship indeterminate (not False, not True)

Not enclosed: ∂(G) < θ_boundary
              → referent does not exist as distinct entity
```

The "zoom out to find encapsulating R" operation — which ROF identifies as
necessary when Sign and Object are not co-enclosed — has a measured cost
in Symphonon:

```
T_local  (referent position)  ≈  0.72
T_global (field average)      ≈  0.088
Fidelity loss from zoom-out:  ~9.6×
```

Calibrating on T_global instead of T_local produces parameters wrong by
an order of magnitude. The loss of fidelity from zooming out is not a
metaphor — it is empirically measured.

---

## 3. Mutability

ROF distinguishes between highly immutable referents (e.g. integers, whose
Sign-Object relationship is stable across contexts) and mutable ones (whose
r(t) changes frequently).

In Symphonon, mutability is a field property, not a referent property:

```
Low mutability:   T_local >> T*  (stable attractor zone)
                  → energy balance strongly positive
                  → large perturbation required to dissolve referent

High mutability:  T_local ≈ T*  (marginal zone)
                  → energy balance near zero
                  → referent forms and dissolves rapidly
```

**Key finding**: the same referent (same internal genome, same ∂(G)) would
be immutable in a high-T attractor zone and highly mutable in a marginal
zone. Mutability is relational, not intrinsic — which is consistent with
ROF's treatment of identity as context-dependent.

---

## 4. Formal vs natural referents

ROF distinguishes:
- Natural referents: immutable rules that exist independently
  (you cannot change how atoms interact through an axiom)
- Formal referents: rules chosen through axioms
  (the system can be self-consistent while not capturing full reality)

The intersection between formal and natural referents shows how well a
formal system models reality — and where it fails.

In Symphonon, T* is a formal parameter calibrated on a natural system
(wind turbine SCADA data, Kelmarsh 2016-2021). The distance between T*
as formally set and T_local as empirically observed is a direct measure
of this intersection:

```
T* (formal parameter)     = 0.601
T_local (natural, mean)   = 0.725
T_local (natural, min)    = 0.551

T* sits between T_local_min and T_local_mean —
correctly placed by calibration, but not derivable
from the formal structure alone.
```

This is a concrete instance of ROF's formal/natural distinction:
the formal parameter T* approaches but does not fully capture the
natural distribution of T_local.

---

## 5. Open question — joint

ROF provides the grammar of what a referent is (structure, identity,
enclosure, evaluability). Symphonon provides the dynamics of what a
referent does (emergence, persistence, dissolution, mutability).

**What neither framework currently provides:**

A theory of why T_local/T_global ≈ 9.6×.

This ratio is empirically stable across all simulation runs. It reflects
the fact that referents actively navigate toward field maxima — they are
not passive objects in a uniform field. But the ratio itself is not derived
from either ROF's axioms or Symphonon's equations. It emerges.

A formal account of this ratio would require:
- ROF side: a theory of how referents select their referential space
  (why they sit at density maxima, not at average density)
- Symphonon side: a derivation of the equilibrium density ratio from
  the navigation dynamics and field equations

This is the joint research question.

---

## Summary

| Concept (ROF)            | Formal correlate (Symphonon) | Empirically measured? |
|--------------------------|------------------------------|-----------------------|
| Referential space R      | Field (φ, ψ)                 | ✓ (field equations)   |
| Membrane / identity      | ∂(G) > θ_boundary            | ✓ (simulation)        |
| Volume / possibility     | T_local distribution         | ✓ (log data)          |
| r(t) realised relation   | T_local(t)                   | ✓ (per-frame log)     |
| Enclosure → evaluability | ∂(G) and T_local > T*        | ✓ (5225+ events)      |
| Zoom-out fidelity loss   | T_local / T_global ≈ 9.6×    | ✓ (empirical)         |
| Mutability               | T_local relative to T*       | ✓ (simulation)        |
| Formal vs natural        | T* vs T_local distribution   | ✓ (calibration data)  |
| Why T_local/T_global≈9.6 | Unknown                      | ✗ (open question)     |

---

*This note is a working document. All claims marked ✓ are verifiable
from the Symphonon codebase and log data at github.com/Brukino/Symphonon*

*Claims marked ✗ are open research questions shared between ROF and Symphonon.*


---

## 6. Rigidity — a fourth observable (Panerati 2026)

A further contribution to the ROF-Symphonon interface emerged from analysis
of collapse mechanisms. Until now the framework described conditions of
persistence. The following defines conditions of failure — two distinct,
symmetric paths:

(A) Internal over-coherence — closed recursion
    The system becomes self-referentially consistent but decouples from
    the broader field. In dynamical systems terms: pathological attractor.
    In ROF terms: the referent's internal r(t) loop closes without reference
    to R (the external field).

(B) External non-integrability — rigidity under shock
    An incoming perturbation cannot be absorbed. The system lacks the
    flexibility to deconstruct and reorganise. In ROF terms: the enclosure
    condition fails because the referential space cannot expand to include
    the new input.

Both paths produce collapse. The unifying variable is rigidity.

**Formal definition:**

    rigidity(g, t) = 1 - rolling_corr(T_local(g,t), T_global(t), window=W)

    rigidity ≈ 0  →  coupled (T_local tracks T_global) — healthy
    rigidity ≈ 1  →  decoupled (T_local insensitive to T_global) — collapse risk

Rigidity is not an intrinsic property of the referent. It is a relational
property: it measures how well the referent's local dynamics remain coupled
to the global field.

**ROF mapping:**

In ROF terms, rigidity measures whether r(t) remains coherent with R.
A rigid referent has r(t) that is internally self-consistent but
no longer co-varies with the broader referential space R.
This is the condition for false(true): locally valid, globally decoupled.

**Empirical consequence:**

In Symphonon P (wind turbine application), this translates to:

    rigidity(tid, t) = 1 - rolling_corr(P_tid(t), P_fleet_mean(t), window=W)

A turbine that decouples from the fleet signal while the fleet remains
coupled is exhibiting rigidity — a measurable early warning of impending
fault. This is testable on Kelmarsh data (6 turbines × 6 years).

**Updated summary table:**

  Concept (ROF)            Symphonon correlate         Empirically measured?
  -----------------------------------------------------------------------
  Referential space R      Field (phi, psi)             yes (field equations)
  Membrane / identity      partial(G) > theta           yes (simulation)
  Volume / possibility     T_local distribution         yes (log data)
  r(t) realised relation   T_local(t)                   yes (per-frame log)
  Enclosure evaluability   partial(G) and T_local > T*  yes (5225+ events)
  Zoom-out fidelity loss   T_local / T_global ~ 9.6x    yes (empirical)
  Mutability               T_local relative to T*       yes (simulation)
  Formal vs natural        T* vs T_local distribution   yes (calibration)
  Rigidity / decoupling    1 - corr(T_local, T_global)  testable (Kelmarsh)
  Why T_local/T_global~9.6 Unknown                      open question

