**Computer Vision Engineer**

*Take-home exercise: sub-pixel spot localization under noise*

# **Context**

We have a pair of systems with cameras and gimbalised lasers on each. Both platforms randomly vibrate. The goal of the system is to accurately point the laser to the terminal of the other side. A camera watches a laser land on its sensor, and we need that spot’s position to be a fraction of a pixel, many times a second (\~1kHz), as the input to a real-time stabilization loop. The spot can be bright or dim depending on the camera settings and environment conditions. The sensor is noisy. Lens behaviour, exposure, gain, temperature and background all move around. Getting a sub pixel precise and real time 1kHz position estimate out of that is most of the job. The end goal for the systems is to point their laser beams precisely towards each other. The output from the camera controls the gimbal that the laser sits on.

This exercise is a simplified stand-in for that problem. You may use AI tools freely. You should understand and be able to defend every design decision in a short live review after you submit.

# **The core task**

This is the baseline we expect a complete submission to cover.

1. **Simulator.** Build an image generator that renders a spot (a Gaussian PSF is fine) at a *known* sub-pixel position, with configurable noise: photon (Poisson) noise, Gaussian read noise, dark current and hot pixels, a non-uniform background gradient, pixel-gain non-uniformity, and bit-depth quantization. Expose an SNR control so you can sweep it. Assume the spot size is \~7 pixels in diameter (1/e^2) but can slightly vary depending on optics quality.

2. **Estimators.** Implement at least two centroiding methods, for example a windowed intensity-weighted centroid with background subtraction, and a 2D Gaussian least-squares or maximum-likelihood fit. Be ready to say why each behaves the way it does.

3. **Characterization.** Using Monte Carlo, measure the **bias** and the **standard deviation** of the position error as a function of SNR for each method. Produce error-versus-SNR plots. State the theoretical precision floor you are comparing against and how close you get to it. Explain which method wins in which regime, and why.

4. **Real-time.** Report per-frame compute cost. State which methods fit a real-time loop at the stated frame rate, and the tradeoff you would make.

# **Dynamic tracking (build your own)**

Extend your simulator to render a *moving* spot over a sequence of frames: a trajectory made of slow drift, random jitter, and one periodic disturbance. Use realistic assumptions that you think may cause these errors and justify them. Because you generated it, you know the true trajectory. Recover the sub-pixel trajectory from the noisy frames, then:

* measure how well you recover it, as position error over the sequence

* identify the periodic disturbance’s frequency and amplitude using time or frequency-domain analysis

* report how close your recovered disturbance is to the values you injected

Design the scenario to be challenging. Identify failure modes. The parameters you pick (the SNR, the disturbance amplitude relative to the jitter, how close the periodic frequency sits to the noise floor) tell us as much as the recovery itself.

# **Real-world conditions**

Our real deployment is outdoors and uncontrolled, not a clean lab. Part of this exercise is thinking about what that changes. Identify the real-world conditions that would degrade a system like this, reason about how each one shows up in the image and what it does to your estimate, and describe how you would detect and handle it. We care first about the quality and range of your thinking here. Actually implementing robustness to any of the conditions you raise is welcome, but the analysis is what matters most. An example of such random fluctuations that often happens is scintillation. Think about how you could include the impact of scintillation in your work and how you could make your system robust against it. 

# **If you want to go further**

The core above is the baseline, not a ceiling. If you want to push this further, we are interested in how far you take it and where you choose to spend the effort. Directions that interest us, none of them required:

* automatic exposure and gain control, with graceful handling of saturation across brightness changes (day / night transitions etc)

* building robustness to the real-world conditions you identified above

* bias frames, flat-field or lens-distortion calibration, and its measurable effect on precision

* holding accuracy under motion blur, or at very low photon counts

* deriving the precision limit from first principles rather than measuring it empirically

* budgeting latency across the full path from photon to position estimate, and where this algorithm sits in it

And anything else you think matters that we did not list. Surprising us is allowed.

# **What a strong submission shows**

* results that are quantified and compared, not asserted: numbers with error bars, methods measured against each other and against a theoretical reference

* clear reasoning about noise, bias, and the limits of what is achievable, not only code that runs

* documentation someone else could pick up and build on: what you built, why, how to reproduce it, what you tested, where it breaks, and what you would do with more time

* honest treatment of assumptions and failure modes

# **Deliverables**

* a runnable repository, where one command reproduces your figures and results

* a written report of your work

* your dynamic scenario, recovered trajectory, and disturbance analysis, reproducible from the repo

# **Ground rules**

* no page limit and no time cap; use your judgment on scope and depth

* ask us if something is genuinely blocking, but expect to make and state your own assumptions

* use AI tools if they help; you must understand and be able to defend what you submit in the live review

We look forward to seeing what you build.