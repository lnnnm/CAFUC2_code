=============================================================================
A High-Fidelity Multi-Model Benchmark Dataset for General Aviation
=============================================================================

Authors: Jing Lu, Yali Fang, Songhan Fan, Xuan Wu, Ziyi Huang
Affiliation: Civil Aviation Flight University of China (CAFUC)
Contact: fyl@cafuc.edu.cn (Yali Fang)
License: Creative Commons Attribution 4.0 International (CC BY 4.0)

1. OVERVIEW
-----------------------------------------------------------------------------
The CAFUC2 dataset is a high-fidelity multivariate time-series benchmark 
designed for General Aviation (GA) anomaly detection. It contains approx. 
1 million data points (sampled at 1 Hz) derived from 120 real flight sorties.
The fleet includes four heterogeneous aircraft models: Cessna 172R, 
Cessna 172S, Cirrus SR20, and Cirrus SR20 G6. 

To provide robust ground-truth labels for safety-critical events, we 
applied a Physics-Based Fault Injection framework to generate four types of 
anomalies, ensuring thermodynamic lag and kinematic consistency.

2. DIRECTORY STRUCTURE
-----------------------------------------------------------------------------
The dataset is organized under the root directory `CAFUC2/` as follows:

CAFUC2/
├── normal_data/                   # Cleaned baseline flight logs
│   ├── C172R/                     # Categorized by aircraft model
│   ├── C172S/
│   ├── SR20/
│   └── SR20G6/           
│       └── cleaned_XXXX_filled.csv  # XXXX: 4-digit flight ID
│
├── abnormal_data/                 # Flight logs with synthetic faults
│   ├── accelerator_operation/     # Throttle Surge (RPM instability & thermal lag)
│   ├── course_deviation/          # Flight Path Deviation (Kinematic lateral drift)
│   ├── engine_power_loss/         # Engine Cooling Failure (Temperature rise)
│   └── pitch_attitude/            # Pitch Excursion (Aerodynamic G-force coupling)
│       └── clean_data_[Model]/
│           └── X.csv              # X: Extracted integer ID corresponding to baseline
│


3. FILE CORRESPONDENCE (COUNTERFACTUAL MAPPING)
-----------------------------------------------------------------------------
To facilitate counterfactual analysis (i.e., comparing a normal flight segment 
against its anomalous counterpart), a strict one-to-one mapping is maintained.

Example: 
- Baseline file: `normal_data/C172R/cleaned_0001_filled.csv`
- Anomalous file: `abnormal_data/pitch_attitude/clean_data_C172R/1.csv`
Both files represent the exact same flight sortie, but the latter contains 
the injected "Pitch Excursion" anomaly at specific timestamps.

4. DATA DICTIONARY (COLUMN DESCRIPTIONS)
-----------------------------------------------------------------------------
Each CSV file contains multivariate time-series flight parameters. Missing 
values in the raw data were interpolated, and sensor noise was smoothed using 
a rolling IQR method. 

To optimize memory footprint and ensure precision, geospatial coordinates 
(Latitude/Longitude) are rounded to 7 decimal places, while all other sensor 
readings are rounded to 3 decimal places.

Variables included (may vary slightly depending on aircraft model avionics):
* label      : Ground truth label (0 = Normal, 1 = Anomaly). Note: Baseline files implicitly have label=0.
* Latitude   : Aircraft latitude coordinate (Decimal Degrees).
* Longitude  : Aircraft longitude coordinate (Decimal Degrees).
* AltMSL     : Altitude above Mean Sea Level (Feet).
* AltGPS     : GPS-derived Altitude (Feet).
* VSpd       : Vertical Speed / Climb rate (Feet per minute).
* Pitch      : Pitch attitude angle (Degrees, positive = nose up).
* Roll       : Roll attitude angle (Degrees, positive = right bank).
* HDG        : Magnetic Heading (Degrees).
* TRK        : Ground Track angle (Degrees).
* NormAc     : Normal Acceleration / G-force (G).
* LatAc      : Lateral Acceleration (G).
* E1 RPM     : Engine 1 Speed (Revolutions per minute).
* E1 FFlow   : Engine 1 Fuel Flow (Gallons per hour).
* E1 OilT    : Engine 1 Oil Temperature (Degrees Fahrenheit).
* E1 CHT1-4  : Engine 1 Cylinder Head Temperature for Cylinders 1 to 4 (Degrees Fahrenheit).
* E1 EGT1-4  : Engine 1 Exhaust Gas Temperature for Cylinders 1 to 4 (Degrees Fahrenheit).
* WptDst     : Distance to the next active waypoint (Nautical miles / Meters).

5. ANOMALY DESCRIPTIONS
-----------------------------------------------------------------------------
1) Throttle Surge (accelerator_operation): 
   Simulates uncommanded engine RPM fluctuations. Fuel flow reacts instantly, 
   while EGT/CHT exhibit a delayed response (thermal lag).
2) Flight Path Deviation (course_deviation): 
   Simulates navigation drift. Geospatial coordinates (Lat/Lon) are 
   recalculated based on deviated track angles using spherical trigonometry.
3) Engine Cooling Failure (engine_power_loss): 
   Simulates cooling system degradation. Causes non-linear CHT temperature 
   rises, coupled with subtle performance drops in Vertical Speed.
4) Pitch Excursion (pitch_attitude): 
   Simulates abrupt attitude upsets. Pitch changes are kinematically 
   coupled with corresponding fluctuations in Normal Acceleration (G-force).

6. CITATION
-----------------------------------------------------------------------------
If you use this dataset in your research, please cite the corresponding Data 
Descriptor published in Scientific Data (Nature Portfolio). 

@article{CAFUC2_2026,
  title={A High-Fidelity Multi-Model Benchmark Dataset for General Aviation Anomaly Detection Generated via Physics-Based Fault Injection},
  author={Lu, Jing and Fang, Yali and Fan, Songhan and Wu, Xuan and Huang, Ziyi},
  journal={Scientific Data},
  year={2026}
}
=============================================================================
