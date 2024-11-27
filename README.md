# Enhanced Autonomous Vehicle Vision

## Executive Summary


DAWN-WEDGE DRIVE combines real-world and synthetic data for autonomous vehicle perception in adverse weather conditions. By merging DAWN's real-world imagery with WEDGE's synthetic data, we address the critical challenge of reliable object detection across diverse weather scenarios. This dataset enables robust model training while reducing the risks and costs associated with collecting real-world data in dangerous weather conditions.


## Potential Applications


- Autonomous vehicle vision systems
- Weather-robust object detection
- Domain adaptation research
- Synthetic-to-real transfer learning
- Safety system development

## Description of Data


Dataset Composition
- DAWN: 1000 real-traffic images, 7,846 objects
- WEDGE: 3,360 synthetic images, 16,513 objects
- Combined: 24,359 annotated objects

## Image Properties


- DAWN Resolution: 1280x675 pixels (864,000 pixels)
- WEDGE Resolution: 640x640 pixels (409,600 pixels)
- XML annotations with standardized format

## Class Distribution


DAWN:
- Cars: 82.3%
- Trucks: 8.2%
- Persons: 6.1%
- Buses: 2.1%
- Other: 1.3%


WEDGE:
- Cars: 40.4%
- Persons: 36.5%
- Trucks: 14.5%
- Buses: 6.9%
- Other: 1.7%


## Power Analysis Results


The combined dataset provides:
- 24,359 total annotated objects
- Coverage across 20 weather conditions
- Balanced class representation through synthetic data
- Diverse object sizes and positions (shown in hexbin plots)
- Complementary resolution characteristics


## Exploratory Data Analysis
Object class distribution
![image](https://github.com/user-attachments/assets/c596e776-6802-4645-a365-7b5721973149)


DAWN Dataset:
- Highly skewed towards cars across all weather conditions
- Consistent pattern across fog, rain, sand, and snow
- Limited representation of smaller objects like bicycles and motorcycles


WEDGE Dataset:
- More balanced distribution across object classes
- Varied object counts across different weather conditions
- Better representation of persons and trucks
- Consistent presence of multiple object classes across weather types


Bounding Box distribution
![image](https://github.com/user-attachments/assets/e4a1ef7f-a2d6-46dc-8b54-af8c230fc779)
![image](https://github.com/user-attachments/assets/2ca5a3b0-d76e-4cd8-8774-66d8b0336178)
![image](https://github.com/user-attachments/assets/05f5e5ef-814c-40ca-a436-c246f07d206f)


- Strong concentration of objects in lower dimensions (0-200 pixels)
- Weather-specific patterns:
- Day conditions show wider spatial distribution
- Night scenes have concentrated clusters in lower regions
- Tornado and hurricane conditions show distinct clustering patterns
- Spring and summer scenes display more dispersed object locations

Aspect Ratio Distribution
![image](https://github.com/user-attachments/assets/9e6cecfb-e246-49f6-9153-ef56307e7235)
- Most objects across weather conditions have aspect ratios between 0-2
- Peak density occurs around aspect ratio of 0.5-1.0
- Long tail distribution extending to aspect ratio of 5
- Different weather conditions show similar aspect ratio patterns


Weather Condition Coverage


- DAWN focuses on four primary conditions with similar object counts
- WEDGE provides broader coverage including seasonal and time-of-day variations
- Different density patterns in hexbin plots suggest varying object detection challenges across weather conditions

## Resolution Analysis
- Bimodal distribution showing clear separation between datasets
- DAWN: Higher resolution for detailed feature capture
- WEDGE: Standardized resolution for consistent processing

## Class Balance
- DAWN shows strong bias toward cars
- WEDGE provides more balanced class distribution
- Combined dataset offers improved representation across classes

## Ethics Statement

- Ensures privacy in real-world data
- Addresses class imbalance and bias
- Promotes safer data collection through synthetic data
- Restricted to research purposes
- Transparent documentation of limitations

## License
Combined dataset: MIT

Link to dataset: https://www.kaggle.com/datasets/supernova5000/enhanced-autonomous-vehicle-vision/
