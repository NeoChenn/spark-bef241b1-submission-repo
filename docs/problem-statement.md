# Problem statement: autonomous flange-integrity robot for offshore platforms

*Working title. Agentverse 2026, Robotics Track. Draft for the technical summary.*

## The user

An offshore maintenance supervisor responsible for the bolted flange connections on a production platform. Their crew must inspect and re-tighten these connections on a schedule, often in gas-classified areas, at height, or next to live equipment.

> **[Team: add one human story here, in one or two sentences. Judges respond to a named person.]**

## The problem

**The work is dangerous.** In 2024 the US mining, quarrying and oil and gas extraction industry had a fatal work injury rate of **13.8 per 100,000 full-time workers**. That is about 6 times manufacturing (2.4), 1.5 times construction (9.2) and nearly 20 times education and health services (0.7). (BLS, Census of Fatal Occupational Injuries, 2024. US data, mostly onshore.)

**The causes are mixed, and we do not claim to fix them all.**
- In one study of 128 offshore oil and gas fatalities, transportation events (helicopters and vessels) were the leading cause: 65 deaths, 51%. (PubMed 23615672. **[Confirm the study period and region.]**)
- In US onshore oil and gas extraction, 2003 to 2006, explosions caused 9% of deaths, falls to a lower level 7% and fires 7%. (CDC MMWR.)

**Bolted flanges are a known weak point.** The US Bureau of Safety and Environmental Enforcement (BSEE) funded research because bolt loosening on flanged connections "poses ongoing safety and reliability risks." Loosened bolts can lead to leaks, and a leak on a platform is a precursor to fire and explosion. Checking and re-tightening them puts people in exactly the zones where a release is most dangerous.

## The gap

Existing work that we found focuses on **subsea** inspection: a BSEE-funded University of Houston project built an ROV that locates, grasps and checks flanges, with over 90% bolt-tightness classification accuracy for most flange sizes. **[We did not find an autonomous topside system that both inspects and corrects. Verify this before we claim it.]**

## What we address, and what we do not

| We address | We do not address |
|---|---|
| Routine inspection and re-torque of flange bolts | Helicopter and vessel transport, the largest single cause in the offshore study |
| Time people spend in gas zones, at height and near live equipment | Well control and emergency response |
| A leak precursor (loosening bolts) | Hot work such as welding. It needs permits and flammable-gas monitoring, so we exclude it (OSHA) |

**Hypothesis, not a result:** if a robot takes over routine tasks, crews can be smaller and make fewer transits, which would cut exposure to the biggest single cause. We state this as a hypothesis.

## Our approach

A robot patrols the platform. It reads **torque stripes** (a paint mark across bolt and flange that breaks or shifts when the bolt turns) **[confirm as industry practice with a source]** and a gas sensor. It decides which bolts need work, aligns a torque tool and re-tightens them. It re-tightens **on its own, without asking for approval**. The safeguards are deterministic and not learned: a **gas-detector interlock** that stops work and retreats, a **diagnosis check** (only a loose bolt is ever tightened; a crack or burst is never touched), and a **verification step** that confirms the stripe aligned and the gas fell. The supervisor is alerted after every repair, and immediately for anything the robot will not handle.

## How we will measure it (in simulation)

- Time to detect a loosened bolt, and missed-bolt rate.
- Minutes a human would otherwise spend in hazard zones, against a manual baseline.
- Correct safe-stop behaviour when a gas alarm, a leak or a person appears mid-task.

## Honest limits

Simulation only. Perception is trained on synthetic data, so the gap to real conditions is untested. The tool alignment uses a simplified constraint rather than full contact physics. Fatality statistics come from the US and are mostly onshore.

---

### Short version for the one-page technical summary (about 100 words)

> Offshore maintenance supervisors send crews to inspect and re-tighten bolted flanges in gas zones, at height and beside live equipment. US oil and gas extraction has a fatal injury rate of 13.8 per 100,000 workers, about 6 times manufacturing (BLS, 2024). Loosening flange bolts are a recognised leak precursor (BSEE). Our robot reads torque stripes and gas readings, decides which bolts need work, and tightens them on its own behind a gas interlock and a diagnosis check, then reports to the supervisor. It does not weld, respond to emergencies or fix transport risk, the largest single cause in one offshore study. We test it in simulation only.

---

### Sources

- BLS, [Number and rate of fatal work injuries by selected private industries, 2024](https://www.bls.gov/charts/census-of-fatal-occupational-injuries/number-and-rate-of-fatal-work-injuries-by-industry.htm)
- CDC MMWR, [Fatalities among oil and gas extraction workers, United States, 2003 to 2006](https://www.cdc.gov/mmwr/preview/mmwrhtml/mm5716a3.htm)
- PubMed, [Fatal injuries in offshore oil and gas operations](https://pubmed.ncbi.nlm.nih.gov/23615672/)
- BSEE, [Robotic assistive smart touch inspection of offshore oil and gas pipelines](https://www.bsee.gov/robotic-assistive-smart-touch-inspection-of-offshore-oil-and-gas-pipelines)
- OSHA, [Oil and gas well drilling and servicing eTool: hot work](https://www.osha.gov/etools/oil-and-gas/general-safety/hot-work-welding)
