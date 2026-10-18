# XAI in Predictive Maintenance — Paper Content

---

## III. Methodology

### A. Simulation and Dataset Generation

The dataset used in this work was generated through a physics-based simulation environment developed in MATLAB/Simulink, modelling a hydraulic machine system susceptible to three primary fault modes: fluid leakage (LeakFault), flow path blockage (BlockingFault), and mechanical bearing degradation (BearingFault). Eight fault configurations were simulated, encompassing all single-fault conditions, all pairwise combinations, and the simultaneous occurrence of all three faults, in addition to a healthy baseline — yielding fault classes 0 through 7. Each configuration was simulated as a complete machine lifecycle, producing time-series observations indexed by operating day.

From each simulated cycle, fourteen features were extracted per observation, spanning both frequency-domain and statistical signal characteristics. Frequency-domain features — peak frequency (*fPeak*), spectral power in low, mid, and high bands (*pLow*, *pMid*, *pHigh*), and spectral kurtosis (*pKurtosis*) — capture the harmonic signature of developing faults. Flow-signal statistics — mean, variance, skewness, kurtosis, peak-to-peak amplitude, crest factor, RMS, mean absolute deviation, and cumulative sum range (*qMean* through *qCSRange*) — characterise distributional shifts in the flow response over the degradation lifecycle.

Each observation is additionally labelled with three continuous severity values corresponding to the three fault types, and an integer fault class label. The final assembled dataset comprises **4,265 samples** distributed approximately uniformly across all eight classes (≈500–555 samples per class), ensuring balanced representation for classifier training. All features were standardised using a StandardScaler fitted exclusively on the training partition to prevent data leakage.

---

### B. Models and RUL Prediction

The predictive framework employs two separate Random Forest models operating in sequence. The first is a multi-class **RandomForestClassifier** (100 estimators, balanced class weights) trained to identify the dominant fault configuration from the 14 extracted features, producing one of eight possible fault class labels. Majority voting across all observations in an uploaded lifecycle determines the final predicted fault class.

The second is a multi-output **RandomForestRegressor** (100 estimators) trained to simultaneously predict the instantaneous severity of all three fault types from the same feature vector. This multi-output formulation allows the model to capture inter-fault correlations in degradation dynamics. The regressor achieves an overall coefficient of determination **R² = 0.968** across the three output dimensions on the held-out test set.

Remaining Useful Life (RUL) is estimated by fitting a power-law degradation model to the regressor's predicted severity trajectory for each active fault. For a severity signal $s(d)$ evolving over operating days $d$, the normalised degradation is modelled as:

$$\tilde{s}(d) = \left(\frac{d}{L}\right)^{\alpha}$$

where $L$ is the predicted lifecycle length (days to end-of-life) and $\alpha$ is the degradation rate exponent. Parameters $L$ and $\alpha$ are estimated via non-linear least squares curve fitting over the observed history up to the current operating day, with $\alpha$ constrained to $[1.0,\ 4.0]$ and $L$ bounded below by the current day to ensure physical validity. The severity signal is smoothed with a three-point uniform filter prior to fitting to reduce the influence of observation noise.

For fault classes involving multiple active severities, RUL is computed independently for each active fault component and the minimum predicted lifecycle — the **worst-case fault** — determines the final reported RUL. This conservative strategy ensures the system flags the most imminent failure regardless of which fault component reaches end-of-life first.

---

### C. SHAP-Based Explainability

To provide interpretable explanations for the regressor's severity predictions, we employ **SHAP (SHapley Additive exPlanations) TreeExplainer**, which computes exact Shapley values for tree-based models in polynomial time. Explanations are generated at the lifecycle level: SHAP values are computed across all rows in the uploaded dataset and aggregated by taking the mean signed value per feature, yielding a direction-aware importance score that reflects how each feature consistently influenced the severity predictions across the entire operating cycle.

A key design consideration is the treatment of fault-specific degradation directionality. BlockingFault severity exhibits a **decreasing** trend over the lifecycle (from an initial value of approximately 0.80 toward 0.53 at end-of-life), as progressive blockage reduces the measured flow signal. LeakFault and BearingFault severities, by contrast, **increase** monotonically as faults worsen. Accordingly, the explanation system filters SHAP contributions by degradation direction:

- For **increasing-severity faults** (LeakFault, BearingFault): only features with positive mean SHAP values are reported, as these drive fault progression.
- For **BlockingFault**: only features with negative mean SHAP values are surfaced, as these indicate worsening blockage.

This directional filtering ensures that reported features are physically meaningful indicators of degradation rather than an undifferentiated mix of positive and negative influences.

For fault classes with multiple active components, independent SHAP explanations are generated for each active severity output. The feature importance percentage for feature $i$ within output $k$ is computed as:

$$p_i^{(k)} = \frac{\overline{|\phi_i^{(k)}|}}{\sum_{j=1}^{F} \overline{|\phi_j^{(k)}|}} \times 100$$

where $\phi_i^{(k)}$ denotes the SHAP value of feature $i$ for output $k$, $\overline{(\cdot)}$ denotes the mean over all lifecycle observations, and $F = 14$ is the number of features. This formulation is **scale-invariant**, making percentages directly comparable across fault types with vastly different severity magnitudes (e.g., LeakFault at $10^{-9}$–$10^{-6}$ versus BlockingFault at $0.53$–$0.80$).

When selecting which severity output to prioritise in multi-fault scenarios, outputs are ranked by a normalised SHAP signal:

$$\text{signal}^{(k)} = \frac{\overline{|\phi^{(k)}|}}{\max(\text{pred}^{(k)}) - \min(\text{pred}^{(k)})}$$

This normalisation prevents outputs with larger absolute scales from dominating the selection purely due to their magnitude.

---

## IV. Results and Discussion

The RandomForestClassifier achieved an overall accuracy of **85.7%** on the held-out test set. Per-class performance was strong across most fault configurations, with F1-scores of **0.91** for LeakFault, **0.92** for BlockingFault, and **0.97** for the Leak+Block compound fault. The weakest performance was observed for Bearing+Leak (F1 ≈ 0.75), reflecting the increased difficulty of discriminating combined fault signatures when two independently weak spectral signals overlap. The balanced class weight scheme proved effective in preventing the classifier from being dominated by any single fault type.

The multi-output regressor demonstrated strong predictive fidelity with an overall **R² = 0.968**. The BlockingFault output, exhibiting the largest absolute severity range (0.53–0.80), contributed most substantially to this aggregate metric. The LeakFault and BearingFault outputs, operating at considerably smaller absolute scales ($10^{-9}$–$10^{-6}$ and $10^{-4}$ respectively), were nonetheless learned with sufficient fidelity for SHAP to produce meaningful, non-degenerate attribution values — confirmed by verifying that per-feature SHAP percentages spanned a wide distribution rather than collapsing to near-uniform contributions.

Power-law RUL estimation produced physically consistent degradation curves for all tested lifecycle inputs. The fitted exponent $\alpha$ remained within the prescribed bounds in all cases, and the extrapolated end-of-life day $L$ provided informative estimates that were validated against the ground-truth total lifecycle length present in each CSV. The interactive validation suite overlays predicted severity curves on ground-truth values and quantifies the RUL error directly, enabling rapid assessment of prediction quality on any new machine lifecycle.

SHAP analysis consistently identified a coherent set of features as primary degradation drivers. **Flow Variance (*qVar*)** and **Low-frequency Power (*pLow*)** emerged as dominant contributors to LeakFault severity progression, reflecting increased low-frequency spectral energy and flow fluctuation associated with developing leakage. For BlockingFault, **Mid-frequency Power (*pMid*)** was the strongest driver, consistent with the known signature of partial flow restriction attenuating mid-band pressure oscillations. BearingFault attribution highlighted **Flow RMS (*qRMS*)** and **Flow Mean Absolute Deviation (*qMAD*)**, indicating that bearing degradation manifests primarily as elevated broadband signal energy rather than narrowband spectral shifts. These attribution patterns align with physical expectations and produce explanations accessible to maintenance engineers without requiring knowledge of machine learning internals.

---

## V. Conclusion

This work presented an end-to-end predictive maintenance framework integrating physics-based fault simulation, multi-output machine learning, power-law RUL estimation, and SHAP-based explainability, delivered through a fully functional interactive web dashboard. The system demonstrated accurate fault classification across eight fault configurations (85.7% accuracy) and high-fidelity severity regression (R² = 0.968), with the combined pipeline providing not only a quantitative RUL estimate but also a transparent, physically grounded explanation of which sensor features drove each prediction.

A central contribution is the **directional SHAP filtering strategy**, which surfaces only degradation-relevant feature attributions aligned with the physical direction of each fault's severity progression — increasing for LeakFault and BearingFault, decreasing for BlockingFault. This avoids the ambiguity of undifferentiated attribution reporting and produces explanations that are immediately actionable for maintenance personnel. Scale-invariant SHAP signal normalisation further ensures that all fault types, regardless of their absolute severity magnitude, receive equitable and meaningful treatment in the explanation pipeline.

The deployed dashboard consolidates the full workflow — fault simulation, dataset upload, RUL prediction, validation against ground truth, and feature attribution — into a single interface, making the system practically usable without requiring specialised expertise in either machine learning or signal processing.

---

## VI. Future Scope

The current system is fully operational and provides accurate fault detection, RUL prediction, and SHAP-based explanations for simulated hydraulic machine lifecycles. Several directions could further extend its capability:

- **Uncertainty quantification**: Augmenting the power-law RUL estimate with prediction intervals (e.g., via conformal prediction or bootstrap resampling of the Random Forest) would provide confidence bounds alongside the point estimate, supporting more risk-aware maintenance scheduling.

- **Real sensor data integration**: The pipeline is designed to accept any conforming CSV input; validating it against real-world sensor data from physical test rigs would assess the sim-to-real transfer gap and guide targeted retraining.

- **Fleet-level monitoring**: Extending the dashboard to track multiple machines simultaneously would enable cross-machine comparison of degradation rates and early identification of systematic failure trends across a fleet.

- **Temporal modelling**: The current per-observation inference treats each day independently. Incorporating sequence models (e.g., LSTM or Temporal Convolutional Networks) could exploit the temporal ordering of lifecycle observations to improve both classification accuracy and severity trend prediction.
