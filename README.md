# **STRUCTURAL VAR**

This project extends the results of [*"Uncertainty Across Volatility Regimes"*]( https://doi.org/10.1002/jae.2672)  to the post-COVID-19 period using identification-through-heteroskedasticity SVAR analysis.

Using a nonrecursive identification scheme for uncertainty shocks that exploits breaks in the volatility of macroeconomic variables, this work addresses two key research questions:  
- **Is uncertainty a cause or an effect of declines in economic activity?**  
- **Does the relationship between uncertainty and economic activity change across macroeconomic regimes?**

---

## **Replication Files for the Paper**

**"Causes and Effects of Business Cycles: An Updated SVAR Identification Through Heteroskedasticity Using Uncertainty Data"**

### **Authors:**  
- **Davide Delfino**, University of Bologna  
- **Giovanni Nannini**, University of Bologna  

---

## **Overview**

This project replicates and extends the analysis of [*"Uncertainty Across Volatility Regimes"*]( https://doi.org/10.1002/jae.2672)  
by Giovanni Angelini, Emanuele Bacchiocchi, Giovanni Caggiano, and Luca Fanelli.  

The contribution extends the observation period up to **2024**, studying the effects of uncertainty even after the **COVID-19 pandemic** and introducing additional identification restrictions for this updated volatility regime. 

### **Key Findings:**  
- **Uncertainty** is more likely a **cause** of declines in economic activity.  
- **Financial and Macroeconomic uncertainty** need to be studied separatelly as uncertainty sources exhibits different dynamics.
- **COVID-19** represent a specific kind of exogenous event. During this period both sources of uncertainty affect on impact the business cycle, but on the medium run are affected endogenously by the economic downturn.

### **Data Sources:**  
The data used in this project were retrieved from:  
- [**FRED: Idustrial Production Total Index**](https://fred.stlouisfed.org/series/INDPRO) for the industrial production growth.
- [**Jurado, Kyle, Sydney C. Ludvigson, and Serena Ng. 2015. "Measuring Uncertainty."**](https://www.sydneyludvigson.com/macro-and-financial-uncertainty-indexes) for the uncertainty measures;
- [**NBER database**](https://www.nber.org/research/business-cycle-dating) for the recession dates;

---

## **Scripts**

### **Main Scripts:**  
- **`Structural_analysis`**: Estimates the on-impact effects for both endogenous and endogenous model and identifies the model that fits better the data. Using the latter model plots the IRF using confidence intervals using a iid bootrsap approach. The time required to run this estimated is estimated to be around 95 seconds.
- **`Preliminary_analysis`**: Implements a descriptive analysis. The time required to run this code is estimated to be around 51 seconds.

### **Supporting Scripts (in `Functions/`):**  
- **`Likelihood_SVAR_%...%`**: Performs all Maximum likelihood estimation needed to run the main scripts.  
