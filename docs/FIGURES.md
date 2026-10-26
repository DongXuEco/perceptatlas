# Manuscript figure guide

Final figure numbering follows the supplied manuscript and supplementary information. Data paths below are found in the data-inclusive release ZIP.

| Figure | Implementation | Source folder | Input status |
|---|---|---|---|
| Fig. 1 | [Fig_1_illustration_functions.py](../code/_support/Fig_1_illustration_functions.py) | `data/Fig_1` | external inputs required |
| Fig. 2 | [Fig_2.py](../code/figures/Fig_2.py) | `data/Fig_2` | analysis tables |
| Fig. 3 | [Fig_3.py](../code/figures/Fig_3.py) | `data/Fig_3` | analysis tables |
| Fig. 4 | [Fig_4.py](../code/figures/Fig_4.py) | `data/Fig_4` | analysis tables |
| Fig. 5 | [Fig_5.py](../code/figures/Fig_5.py) | `data/Fig_5` | analysis tables |
| Fig. 6 | [Fig_6.py](../code/figures/Fig_6.py) | `data/Fig_6` | analysis tables |
| Extended Data Fig. 1 | [Extended_Data_Fig_1.py](../code/figures/Extended_Data_Fig_1.py) | `data/Extended_Data_Fig_1` | analysis tables |
| Extended Data Fig. 2 | [Extended_Data_Fig_2.py](../code/figures/Extended_Data_Fig_2.py) | `data/Extended_Data_Fig_2` | analysis tables |
| Extended Data Fig. 3 | [Extended_Data_Fig_3.py](../code/figures/Extended_Data_Fig_3.py) | `data/Extended_Data_Fig_3` | analysis tables |
| Extended Data Fig. 4 | [Extended_Data_Fig_4.py](../code/figures/Extended_Data_Fig_4.py) | `data/Extended_Data_Fig_4` | analysis tables |
| Extended Data Fig. 5 | [Extended_Data_Fig_5.py](../code/figures/Extended_Data_Fig_5.py) | `data/Extended_Data_Fig_5` | analysis tables |
| Extended Data Fig. 6 | [Extended_Data_Fig_6.py](../code/figures/Extended_Data_Fig_6.py) | `data/Extended_Data_Fig_6` | analysis tables |
| Extended Data Fig. 7 | [Extended_Data_Fig_7.py](../code/figures/Extended_Data_Fig_7.py) | `data/Extended_Data_Fig_7` | external inputs required |
| Supplementary Fig. S1 | [Supplementary_Fig_S1.py](../code/figures/Supplementary_Fig_S1.py) | `data/Supplementary_Fig_S1` | external inputs required |
| Supplementary Fig. S2 | [Supplementary_Fig_S2.py](../code/figures/Supplementary_Fig_S2.py) | `data/Supplementary_Fig_S2` | external inputs required |
| Supplementary Fig. S3 | [Supplementary_Fig_S3.py](../code/figures/Supplementary_Fig_S3.py) | `data/Supplementary_Fig_S3` | external inputs required |
| Supplementary Fig. S4 | [Supplementary_Fig_S4.py](../code/figures/Supplementary_Fig_S4.py) | `data/Supplementary_Fig_S4` | external inputs required |
| Supplementary Fig. S5 | [Supplementary_Fig_S5.py](../code/figures/Supplementary_Fig_S5.py) | `data/Supplementary_Fig_S5` | external inputs required |
| Supplementary Fig. S6 | [Supplementary_Fig_S6.py](../code/figures/Supplementary_Fig_S6.py) | `data/Supplementary_Fig_S6` | external inputs required |
| Supplementary Fig. S7 | [Supplementary_Fig_S7.py](../code/figures/Supplementary_Fig_S7.py) | `data/Supplementary_Fig_S7` | external inputs required |
| Supplementary Fig. S8 | [Supplementary_Fig_S8.py](../code/figures/Supplementary_Fig_S8.py) | `data/Supplementary_Fig_S8` | external walking configuration required |
| Supplementary Fig. S9 | [Supplementary_Fig_S9.py](../code/figures/Supplementary_Fig_S9.py) | `data/Supplementary_Fig_S9` | external inputs required |
| Supplementary Fig. S10 | [Supplementary_Fig_S10.py](../code/figures/Supplementary_Fig_S10.py) | `data/Supplementary_Fig_S10` | analysis tables |
| Supplementary Fig. S11 | [Supplementary_Fig_S11.py](../code/figures/Supplementary_Fig_S11.py) | `data/Supplementary_Fig_S11` | external inputs required |
| Supplementary Fig. S12 | [Supplementary_Fig_S12.py](../code/figures/Supplementary_Fig_S12.py) | `data/Supplementary_Fig_S12` | external inputs required |
| Supplementary Fig. S13 | [Supplementary_Fig_S13.py](../code/figures/Supplementary_Fig_S13.py) | `data/Supplementary_Fig_S13` | external inputs required |

## Figure descriptions

### Fig. 1

Workflow for mapping and analysing urban emotional livability across 5,880 cities.

Illustration helper functions are supplied alongside code/_support/Fig_1_environment_illustrations.py. No complete Fig_1.py entry exists. The final manuscript composite was assembled manually and requires external artwork; street-view images are not included.

### Fig. 2

Global urban emotional perception and its distribution across urban contexts. a-c, Emotional livability index (ELI), positive perception and negative perception across 5,880 cities. d-i, Perceived beauty, wealth, safety, liveliness, boredom and depression. Insets show city level score distributions and medians. j-o, ELI distributions across quintiles of population density, night time lights, green space cover, built-up density, non-motorized service travel time and land surface temperature. Q1-Q5 indicate increasing indicator values. Boxes show interquartile ranges and medians; whiskers span the 5th-95th percentiles. Dashed lines connect group medians. Spearman’s ρ and spatial block bootstrap P values describe city level associations.

Analysis-table figure entry. Related shared inputs are indexed in file_inventory.csv.

### Fig. 3

Within-city perceptual inequality and perceived quality across Global North and South cities. a, North–South distributions of composite Gini coefficients. b, Global distribution of composite Gini, averaged across six perceptual dimensions. c–h, Gini distributions for beauty, wealth, safety, liveliness, boredom and depression, respectively. In a and c–h, curves show empirical quantiles and shading indicates relative city density. Δ denotes the North minus South mean difference; P values are from two-sided Welch’s t-tests with Holm correction across seven comparisons. The interval in a is a 95% bootstrap confidence interval. i, Pearson correlations among inequality metrics; asterisks indicate unadjusted approximate P values from Fisher’s z transformation: *P < 0.05, **P < 0.01 and ***P < 0.001. j, Joint distribution of mean ELI and composite Gini, divided by global medians. k, City percentages in the corresponding categories, with 95% bootstrap confidence intervals.

Analysis-table figure entry. Related shared inputs are indexed in file_inventory.csv.

### Fig. 4

Perceptual disparities between urban cores and expansion areas. a, Global distribution of core–expansion differences in ELI. b, Distributions of core advantage in ELI and six perceptual dimensions. c, Correspondence between core and expansion ELI; the dashed line indicates equal scores, and the shaded band denotes differences within ±0.03. d, Shares of cities with a core advantage in each perceptual dimension, with 95% bootstrap confidence intervals. e, North–South comparison of core–expansion ELI differences; P values are from a two-sided Welch’s t-test. f, Shares of cities with expansion advantage (< −0.03), approximate balance (−0.03 to +0.03) or core advantage (> +0.03), defined by core minus expansion ELI. g, Distributions and mean perception scores in core and expansion areas by North–South group. Open squares and filled circles indicate expansion and core means, respectively; boxes show interquartile ranges with median lines, and whiskers indicate the 5th–95th percentiles. Core advantage denotes higher ELI or positive perception scores, or lower boredom and depression scores.

Analysis-table figure entry. Related shared inputs are indexed in file_inventory.csv.

### Fig. 5

Environmental associations of urban emotional livability and spatial inequality across Global North and South cities. a, Overview linking ELI and three inequality measures to important predictor families and individual variables. b, Relative importance of predictor families for ELI and inequality in Global North and South cities. c, Importance of leading ELI predictors in global, northern and southern models. d, Importance of leading inequality predictors for the Gini coefficient, Theil index and P90-P10 spread. In c,d, importance is measured as the decrease in out-of-fold R2 after predictor permutation; values are means, with standard deviations in parentheses. e, Modelled associations of selected predictors with ELI and Gini inequality. Solid blue lines show ALE, dashed orange lines show centred partial dependence (PDP), and shaded bands indicate 95% bootstrap envelopes. Horizontal axes express predictor values as percentiles. POI, point of interest; LST, land surface temperature; NTL, night time lights.

Analysis-table figure entry. Related shared inputs are indexed in file_inventory.csv.

### Fig. 6

Environmental associations of emotional livability in urban cores and expansion areas across the Global North and South. a, Predictor groups and their importance shares in core and expansion area models, shown by the inner and outer rings, respectively. b, Permutation importance of selected predictors for core and expansion area ELI; values are means, with standard deviations in parentheses. c, Pairwise Spearman correlations among predictors in cores (upper triangle) and expansion areas (lower triangle). d, Relative importance of predictor families in northern cores, northern expansion areas, southern cores and southern expansion areas. e, Nonlinear associations between selected predictors and ELI in cores and expansion areas. Solid blue lines show ALE, and dashed orange lines show centred PDP. Vertical dotted lines and q labels identify the first zero crossing of the centred ALE curve along the plotted predictor gradient. N.C., northern cores; N.E., northern expansion areas; S.C., southern cores; S.E., southern expansion areas.

Analysis-table figure entry. Related shared inputs are indexed in file_inventory.csv.

### Extended Data Fig. 1

Validation and robustness of the ELI. a, Associations between city-level ELI and positive response shares from the European Commission’s Quality of Life survey. Pearson’s r and Spearman’s ρ are reported; P values refer to Spearman tests. b, City-ranking uncertainty estimated from 1,000 normal-approximation simulations, showing 95% rank-interval widths and their moving median. c, Rank correlations with baseline ELI and retention of baseline top- and bottom-decile cities across 500 random weighting schemes. Lines connect results from the same scheme; diamonds indicate medians and whiskers span the 2.5th–97.5th percentiles. d, Sensitivity to alternative index constructions. Coordinates indicate top- and bottom-decile retention, arc completion represents Spearman’s ρ with baseline ELI, and centre colour indicates the Global North minus Global South mean ELI.

Analysis-table figure entry. Related shared inputs are indexed in file_inventory.csv.

### Extended Data Fig. 2

Disparities in urban emotional perception between Global North and South cities. a–c, Quantile distributions of the emotional livability index (ELI), composite positive perception and composite negative perception scores, respectively. d–i, Corresponding distributions for beauty, wealth, safety, liveliness, boredom and depression. Curves show empirical quantiles, and shading indicates relative city density. Orange and blue represent Global South and North cities, respectively. Diamonds and error bars below each panel indicate North minus South mean differences and 95% Welch confidence intervals. P values are from two-sided Welch’s t-tests with Holm correction across the nine comparisons.

Analysis-table figure entry. Related shared inputs are indexed in file_inventory.csv.

### Extended Data Fig. 3

Predictor characteristics, predictive performance and robustness of models of urban emotional perception. a, Classification of the 25 predictors into nine thematic groups and their data coverage at the city and urban zone levels. b, Variance inflation factors (VIFs) for selected predictors at both levels; the dashed line marks VIF = 5. c, Pairwise Spearman correlations among predictors at the city level (lower triangle) and urban zone level (upper triangle). d, Cross validated predictive performance for ELI, perceptual inequality and positive and negative perception scores across geographical groups and urban zones. Points and error bars indicate mean R2 ± standard deviation. e, Predicted versus observed city level ELI (r = 0.90), with the dashed black line indicating 1:1 agreement. f, Stability of the highest ranked predictor family and overlap among the three highest ranked families across analytical checks. The mean overlap among the three highest ranked families was 2.5, while retention of the highest ranked family ranged from 56% to 100% across the displayed checks. g, Correspondence between absolute bivariate Spearman correlations and normalized machine learning predictor importance. Absolute bivariate correlations showed little correspondence with normalized model importance, further distinguishing marginal associations from predictive importance in multivariable models.

Analysis-table figure entry. Related shared inputs are indexed in file_inventory.csv.

### Extended Data Fig. 4

Environmental contrasts between Global North and South cities and between urban cores and expansion areas. a, Environmental variables with the largest absolute North-South contrasts in median robust standardized values. b, Variables with the largest absolute paired core-expansion contrasts. c, North-South differences in core-expansion environmental contrasts. Points and horizontal bars in a-c indicate estimated contrasts and 95% confidence intervals, respectively; dashed vertical lines indicate zero difference. d, Median robust standardized values of selected variables in northern cores, northern expansion areas, southern cores and southern expansion areas. e, Median absolute standardized contrasts across variables within each thematic family for North-South comparisons, core-expansion comparisons and North-South differences in core-expansion contrasts. f, Numbers of variables whose contrast confidence intervals exclude zero, relative to the total number of variables in each family. NS, North-South; C-E, core-expansion; Mod., North-South difference in core-expansion contrasts. Standardized contrasts describe differences in environmental conditions, rather than effects on emotional livability or perceptual inequality.

Analysis-table figure entry. Related shared inputs are indexed in file_inventory.csv.

### Extended Data Fig. 5

Spatial variation in the dominant environmental predictor families of urban emotional livability and within city perceptual inequality. a,b, Geographical distributions of the dominant predictor families for city level emotional livability index (ELI; a) and composite Gini inequality (b). Dominant families are identified by their share of local SHAP attribution magnitude. c, Joint geographical distribution of city level ELI and composite Gini. The bivariate colour scheme distinguishes combinations of higher and lower perceived quality and inequality. d, Agreement between the dominant predictor families for ELI and Gini within each city. e, Percentages of cities assigned to each dominant predictor family under the global models for ELI, Gini, Theil and the P90-P10 spread. f, Distributions across cities of the dominant family share of local SHAP attribution magnitude for each global model. Higher shares indicate that attribution is more concentrated in the leading predictor family.

Analysis-table figure entry. Related shared inputs are indexed in file_inventory.csv.

### Extended Data Fig. 6

Spatial variation in the dominant environmental predictor families of emotional livability across urban cores and expansion areas. a,b, Geographical distributions of the dominant predictor families for emotional livability index (ELI) predictions in urban cores (a) and expansion areas (b). c, Joint geographical distribution of core and expansion area ELI. d, within city differences in the dominant predictor family between cores and expansion areas.

Analysis-table figure entry. Related shared inputs are indexed in file_inventory.csv.

### Extended Data Fig. 7

Global coverage of street view and Landsat imagery and contrasts between the Global North and South. a, Spatial distribution of street view image density, expressed as the number of images per square kilometre. b, Spatial distribution of Landsat-8 image counts. Annotations report the corresponding values for the Global North and South and their North to South ratios.

Full drawing requires local spatial or illustration inputs that are outside this deposit.

### Supplementary Fig. S1

Bivariate spatial patterns of urban emotional perception across six dimensions. Bivariate spatial patterns of urban emotional perception across six dimensions. Lower-triangle panels show pairwise combinations of city-level perception scores globally, whereas upper-triangle panels show the corresponding combinations of raster-based scores within a selected urban area.

Full drawing requires local spatial or illustration inputs that are outside this deposit.

### Supplementary Fig. S2

Validation and spatial comparison of downscaled land surface temperature (LST). a, Summary of internal five fold cross validation and independent holdout validation, reporting sample size, coefficient of determination (R2), root mean square error (RMSE), mean absolute error (MAE), mean bias and the percentage of predictions within 5 K of observed values. b, Residual distributions for the two validation datasets; points and horizontal lines indicate medians and 5th-95th percentiles. c,d, Predicted versus observed LST for five-fold cross-validation (c) and independent validation (d). e, Residuals across the observed LST gradient; lines represent bin wise medians and shaded bands indicate interquartile ranges. f, Bland-Altman diagnostic showing residuals against the mean of predicted and observed LST. The solid line indicates pooled mean bias, and dashed lines indicate the 95% limits of agreement. g, Global distributions of MODIS, downscaled and Landsat LST, with inset pixel density distributions and a shared temperature scale. h, Comparison of MODIS and downscaled LST in Beijing, Los Angeles, Singapore, Rio de Janeiro, Rome and Cape Town, with corresponding Google imagery for spatial context.

Full drawing requires local spatial or illustration inputs that are outside this deposit. Numerical validation records support panels a–f; spatial panels g–h require external inputs.

### Supplementary Fig. S3

Global Landsat 8 median composite after cloud masking, 2018-2024.

Full drawing requires local spatial or illustration inputs that are outside this deposit.

### Supplementary Fig. S4

Global distribution of spectral endmember samples and representative surface examples. a, Geographical distribution of samples representing vegetation, bright substrate and water, shown as green circles, magenta triangles and blue squares, respectively. b-d, Satellite image examples of vegetation (b), bright substrate (c) and water (d) sampling locations.

Full drawing requires local spatial or illustration inputs that are outside this deposit.

### Supplementary Fig. S5

Global distributions and urban patterns of vegetation, bright substrate and water endmember fractions. a-c, Global distributions of vegetation (a), bright substrate (b) and water (c) endmember fractions. d, Ternary composite showing the relative contributions of the three endmembers. e, Spatial patterns of endmember fractions in six representative cities: Beijing, Los Angeles, Singapore, Rio de Janeiro, Rome and Cape Town. Rows show vegetation, bright substrate and water fractions, followed by Google imagery for visual comparison. Endmember maps use the corresponding colour scales in a-c.

Full drawing requires local spatial or illustration inputs that are outside this deposit.

### Supplementary Fig. S6

Global distribution and composition of points of interest across six service categories. a-f, Spatial distributions of points of interest (POIs) classified as daily living services (a), production services (b), transportation services (c), education and culture (d), healthcare (e), and leisure and entertainment (f). Colours represent POI counts per mapping grid cell, as indicated by the shared colour bar. Circular insets report the proportion and number of POIs assigned to each category.

Full drawing requires local spatial or illustration inputs that are outside this deposit.

### Supplementary Fig. S7

Global distribution and composition of the road network by road class. a-p, Spatial distributions of bridle paths (a), cycleways (b), footways (c), pedestrian priority residential streets (d), motorways (e), paths (f), pedestrian streets (g), primary roads (h), residential streets (i), secondary roads (j), service roads (k), stairways (l), tertiary roads (m), agricultural and forestry tracks (n), trunk roads (o), and roads with unknown classes (p). Circular insets summarize the contribution of each class to the mapped road network.

Full drawing requires local spatial or illustration inputs that are outside this deposit.

### Supplementary Fig. S8

Global patterns of estimated walking travel time to six categories of service facilities. a-f, Estimated minimum walking travel times to daily living services (a), production services (b), transportation services (c), education and culture (d), healthcare (e), and leisure and entertainment (f). Annotations report mean travel times for the Global North and South and their North to South ratios. Ratios below one indicate shorter mean travel times in the North.

The shared drawing implementation is provided, but the original walking parameters were not recovered. Supply walking_plot_parameters.json with raster_files (six entries containing file, title, north, south), vmin and vmax. No driving parameters are substituted.

### Supplementary Fig. S9

Global patterns of estimated driving travel time to six categories of service facilities. a-f, Estimated minimum driving travel times to daily living services (a), production services (b), transportation services (c), education and culture (d), healthcare (e), and leisure and entertainment (f). Annotations report mean travel times for the Global North and South and their North to South ratios.

Full drawing requires local spatial or illustration inputs that are outside this deposit.

### Supplementary Fig. S10

Validation of global public service accessibility against Google route planning estimates. a, Geographical distribution of 500 sampled cities and 2,500 validation origins. The inset shows residual distributions for walking and driving, calculated as raster estimated minus Google route planning travel times. b, Overall correspondence between raster estimated and Google travel times for walking (non-motorized) and driving (motorized). c,d, Service specific comparisons for walking (c) and driving (d) across six public service categories: education, entertainment, healthcare, living services, production services and transport. Ful sample validation statistics and filtering procedures are provided in Supplementary Methods S2.

Analysis-table figure entry. Related shared inputs are indexed in file_inventory.csv.

### Supplementary Fig. S11

Spatial layers used to construct the travel time friction surface. a,b, Off road (a) and road (b) travel speed layers. c, Combined travel speed layer. d, Corresponding friction surface, expressing the estimated time required to traverse one metre. Speeds are expressed in kilometres per hour and friction in minutes per metre. Higher friction values indicate slower movement.

Full drawing requires local spatial or illustration inputs that are outside this deposit.

### Supplementary Fig. S12

Modelled service accessibility and travel time friction surfaces in six cities. Rows show Beijing, Los Angeles, Singapore, Rio de Janeiro, Rome and Cape Town. Columns show mean walking travel time, mean driving travel time, walking friction, driving friction and satellite imagery, respectively. Travel times are expressed in minutes and friction values in minutes per metre. Walking and driving travel time maps use different colour scales. Satellite imagery provides geographical context for the modelled spatial patterns.

Full drawing requires local spatial or illustration inputs that are outside this deposit.

### Supplementary Fig. S13

Bivariate spatial patterns of accessibility and service infrastructure. a,b, Joint spatial distributions of accessibility scores and road density (a) or POI density (b). The horizontal axes of the bivariate legends represent infrastructure density, and the vertical axes represent accessibility scores.

Full drawing requires local spatial or illustration inputs that are outside this deposit.

See [reproduction limits](REPRODUCIBILITY.md), [input mapping](../figure_inputs.csv) and [external input requirements](../external_inputs_required.csv).
