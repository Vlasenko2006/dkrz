

Applying for an AI-Consultant position at DKRZ, I was asked to demonstrate my skills in neural networks, and data processing, on the example of raw Earth monitoring satellite data (see description below). What exactly I was supposed to do with this data was not specified, giving me room for imagination. To complete the task I had two days.

## Introduction

The Cyclone Global Navigation Satellite System **[(CyGNSS)](https://en.wikipedia.org/wiki/Cyclone_Global_Navigation_Satellite_System)** is a space-based system developed by the University of Michigan and Southwest Research Institute to improve hurricane forecasting by better understanding the interactions between the sea and the air near the core of a storm. It consists of several micro-satellites flying in low Earth orbit that continuously receive positioning data from GPS/GLONASS satellites through two types of antennas:

- **Zenith antenna**: Receives direct signals from GPS/GLONASS.
- **Side antenna complex**: Captures reflected signals from the Earth's surface.

When ocean waves are present, the reflected signal undergoes a **Doppler shift**, which increases with wave roughness. By comparing the direct and reflected signals, we can determine the Doppler shift and, consequently, estimate wave height and wind strength. Receiving GPS signals CyGNSS satellites transmit the Doppler shifts to the receivers on Earth. 

## The Problem
Since CyGNSS satellites have a relatively short lifetime and are prone to failures, we consider two scenarios:
1. **A satellite is lost**, and we need to estimate its measurements using data from the remaining satellites.
2. **A satellite experiences an instrumentation failure**, and we use data from other satellites to detect anomalies.

### Step 1: Prepare the dataset
The dataset (`.ncl` file from DKRZ) contains measurements from **8 CyGNSS satellites** (numbered 1–8). For the whole time series, I did the following:
- In the data series, I removed the signal for at least one satellite assuming that it is lost.
-  I randomly vary the number of lost satellites, their ID, and the time of the lost signal.
- I used the signals of the *lost* satellites as labels for the neural network, which it should recover.
- I used 75% and 25% of the time series for training and validation.   

### Step 2: Defining Inputs for the Neural Network

Since all CyGNSS satellites follow the same orbit, we fix a **reference point** on this orbit, where we want to reconstruct the data. Consider:
- Satellite **id=0** as the **lost** satellite.
- Satellite **id=1**  which follows behind id=0 in orbit.
- When id=1 reaches the position of id=0, the Earth has rotated by ~200 km.
- Atmospheric phenomena are large-scale ~1000 km, so wind strength does not change much over such distances.
- Thus, satellite **id=1** will measure similar data to satellite **id=0** in the vicinity of the reference point (reconstruction).
- This is also true for a satellite with **id=2**, but since the Earth displacement is ~400 km, the reconstruction area will be smaller.
- For satellites with larger IDs there might be no reconstruction area and they must be expelled from the neural network inputs.  

To reconstruct the missed data we need to determine the IDs of satellites, valid for this purpose and the corresponding reference areas. Note that Satelites move with a constant speed, and we can measure associate the size of the area with the time interval $\tau$, that satelite passes it. We do so by solving the following **maximization problem**:
```math
 \arg\max_{\tau} \sum (v_{0}(t) \cdot v_{i}(t-\tau))
```
Where:
- $$\( v_0(t) \) and \( v_i(t) \)$$ are wind velocity measurements from id=0 and id=i, $i=1,..,8$ at moment $$t$$.
- $$\( \tau \)$$ is the time lag that maximizes similarity.

Computing IDs and corresponding $\tau$s we obtain the data and group of satelites valid for reconstruction.

### Step 3: Neural Network Construction

Since the data has a strong **temporal component**, we use deep learning models suitable for **time series forecasting**:
1. **Recurrent Neural Network (RNN)** – Specialized in sequential data.
2. **1D Convolutional Neural Network (CNN)** – Detects patterns in time series.
3. **Vanilla Neural Network** – Used as a reference, to check how RNN and CNN overperform it.
4. **Multivariate Regression** – A conventional approach for comparison.

## Results

 The performance, measured as correlation between predicted and actual wind speeds for id=0, was:

| Model                | Correlation |
|----------------------|-------------|
| **Recurrent Neural Network** | **0.70** |
| **1D Convolutional Neural Network** | **0.70** |
| Vanilla Neural Network | 0.65 |
| Linear Regression | 0.57 |

### Visualization of Results

**Figure 1**: True (red) vs. Predicted (blue) wind velocities for satellite id=0.

![Sample Output](https://github.com/Vlasenko2006/dkrz/blob/main/Result.jpg)

## Conclusions

- **We can successfully restore missing data** for a lost satellite (id=0) using data from other satellites.
- The best-performing models were **Recurrent Neural Networks (RNNs) and 1D Convolutional Neural Networks**, both achieving a correlation of **0.70**.
- **For anomaly detection**, the same approach can be used, except the time lag \( \tau \) should be negative.

