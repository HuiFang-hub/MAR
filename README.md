# Mar
This repo contains the sample code for reproducing the results of our paper: **Interpretable machine learning enables high performance of magnesium borohydride hydrogen storage system**

## Abstract

Machine learning has considered as an essential and innovative instrument in accelerating high-throughput design and mechanism discovery of energy storage materials. This study proposes, for the first time, a novel machine learning model to predict dehydrogenation behaviors of modified Mg(BH4)2, aiming to provide optimal solutions to the sluggish kinetics of Mg(BH4)2. Notably, numerous datapoints are collected from temperature-programmed, isothermal, cyclic dehydrogenation behaviors, and a neural network model is proposed by using multi-head attention mechanisms, which exhibits highest predictive performance compared to traditional machine learning models. The study also ranks different variables influencing dehydrogenation processes, employing interpretable analysis to identify critical variable thresholds, offering guidance for the experimental parameter design. Our model can also be adapted to scenarios involving co-doping of hydrides and catalysts in Mg(BH4)2 system, and proved high accuracy and scalability in predicting dehydrogenation curves under diverse conditions. Employing the model, predictions for a series of undeveloped Mg(BH4)2 co-doping systems can be made, and superior dehydrogenation catalytic effects of fluorinated graphite (FGi) is uncovered. Real-world experimental validation of the optimal Mg(BH4)2-LiBH4-FGi system confirms consistency with model predictions, and performance enhancement attributes to experimental parameter optimization. Further characterizations provide mechanistic insights into the synergistic interactions of FGi and LiBH4. This work paves the way for advancing utilization of machine learning in high-capacity hydrogen storage field.

## Quick Start

1. Prepare the environments

   ```bash
   pip install -r requirements.txt
   ```

2. Bash run

   ```bash
   bash scripts/main.bash
   ```

   

