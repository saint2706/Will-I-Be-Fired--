# Business Actions: Termination Risk Response Framework

This document maps predicted termination risk scores to specific, actionable HR interventions. The goal is to provide clear guidance on how to respond to model predictions in a way that supports employee retention and well-being.

## Risk Score Bands & Recommended Actions

### Low Risk: 0-10%
**Interpretation:** Employee is highly stable with minimal termination risk.

**Recommended Actions:**
- **Standard Management:** Continue regular 1:1 check-ins at normal cadence (monthly or quarterly)
- **Development Focus:** Identify growth opportunities and career path discussions
- **Recognition:** Acknowledge contributions to maintain engagement
- **Documentation:** No special intervention needed

**Expected Outcome:** Maintain high retention with minimal active intervention.

---

### Low-Moderate Risk: 10-30%
**Interpretation:** Slightly elevated risk; employee may be experiencing minor dissatisfaction or stressors.

**Recommended Actions:**
- **Increased Check-ins:** Move to bi-weekly 1:1s with direct manager
- **Engagement Survey Follow-up:** Review recent engagement survey responses; address specific concerns
- **Workload Assessment:** Evaluate current workload and project assignments for balance
- **Skills Development:** Offer relevant training or professional development opportunities
- **Recognition Programs:** Nominate for peer recognition or spot bonuses

**Expected Outcome:** Early intervention can reduce risk by addressing minor issues before they escalate.

---

### Moderate Risk: 30-60%
**Interpretation:** Moderate termination risk; significant intervention warranted.

**Recommended Actions:**
- **Weekly Manager 1:1s:** Increase touchpoint frequency to weekly for at least 4-6 weeks
- **Compensation Review:** Conduct off-cycle salary review to ensure market competitiveness
- **Career Pathing Session:** Dedicated meeting with manager + HR to discuss career goals and internal mobility
- **Flexible Work Arrangements:** Explore schedule flexibility, remote work options, or compressed workweeks
- **Mentorship/Coaching:** Pair with senior mentor or provide professional coaching services
- **Special Projects:** Assign high-visibility projects aligned with employee interests
- **Benefits Counseling:** Ensure employee is aware of all available benefits (mental health, EAP, etc.)

**Expected Outcome:** Proactive intervention can reduce voluntary terminations by 30-50% in this cohort.

---

### High Risk: 60%+
**Interpretation:** Critical risk; immediate action required.

**Recommended Actions:**
- **Immediate HR Escalation:** Notify HR Business Partner within 24 hours
- **Retention Meeting:** Schedule urgent meeting with manager, HR, and optionally skip-level manager
- **Stay Interview:** Conduct structured stay interview to understand root causes
- **Retention Package:** Consider:
  - Retention bonus (e.g., 10-20% of salary vesting over 12-24 months)
  - Immediate salary adjustment if below market
  - Title/responsibility change
  - Team transfer or role modification
- **Daily/Bi-weekly Check-ins:** Very frequent touchpoints until risk stabilizes
- **Exit Alternative Planning:** If termination seems likely, prepare for knowledge transfer and succession planning
- **Mental Health Support:** Proactive outreach about EAP and mental health resources

**Expected Outcome:** High-touch intervention can save 20-40% of high-risk employees; for others, enables graceful transition.

---

## Expected Impact Calculation

### Assumptions
- **Baseline Turnover Rate:** 33.4% (from historical data)
- **At-Risk Population:** Employees with predicted risk >30%
- **Intervention Success Rates:**
  - Low-Moderate (10-30%): 50% reduction in risk with early intervention
  - Moderate (30-60%): 40% retention improvement
  - High (60%+): 30% retention improvement

### Projected Outcomes
If 100 employees are scored:
- **10-30% band:** ~20 employees → 10 saved with early action
- **30-60% band:** ~15 employees → 6 saved with moderate intervention
- **60%+ band:** ~10 employees → 3 saved with intensive intervention

**Total Projected Retention Improvement:** ~19 employees out of 100 at-risk
**Estimated Cost Avoidance:** 19 × (replacement cost of 1.5-2× annual salary) = significant ROI

### Cost-Benefit Analysis
- **Cost of Interventions:** $2k-$10k per employee (coaching, bonuses, HR time)
- **Cost of Replacement:** $75k-$150k per employee (recruiting, onboarding, lost productivity)
- **Net Benefit:** $50k-$140k per successful retention

---

## Policy Configuration

Risk thresholds and action mappings are configurable via `configs/policy.yaml`. This allows HR teams to adjust interventions based on organizational context and resource availability.

**Key Parameters:**
- `low_threshold`: Upper bound for low-risk band (default: 0.10)
- `moderate_threshold`: Upper bound for low-moderate risk (default: 0.30)
- `high_threshold`: Upper bound for moderate risk (default: 0.60)
- `action_labels`: Human-readable descriptions for each band

---

## Ethical Considerations

1. **Transparency:** Employees should be informed (at least in aggregate) that retention analytics are used.
2. **Non-Punitive:** Predictions should **never** be used to justify termination or reduce opportunities.
3. **Privacy:** Individual scores should be shared only with direct manager and HR on a need-to-know basis.
4. **Bias Monitoring:** Regularly audit actions by demographic subgroups to prevent discriminatory patterns.
5. **Employee Agency:** Interventions should be offered, not mandated; respect employee autonomy.

---

## Usage in Tools

### CLI
```bash
python src/predict_cli.py --model models/best_model.joblib --calibrate
```
Displays risk band and suggested actions alongside probability scores.

### GUI
The Streamlit dashboard includes a dedicated "Business Actions" panel that:
- Shows the employee's risk band
- Lists specific recommended actions
- Provides downloadable action plans for managers
- Allows customization of thresholds via sidebar

---

## Revision History

- **v1.0 (2025-11):** Initial framework based on ML model deployment
- Future revisions will incorporate feedback from HR pilots and A/B testing of interventions
