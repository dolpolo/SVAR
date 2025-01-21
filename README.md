# **STRUCTURAL VAR**

## [**Extending Results of "Uncertainty Across Volatility Regimes"**]( https://doi.org/10.1002/jae.2672)  

This project extends the results of *"Uncertainty Across Volatility Regimes"* to the post-COVID-19 period using identification-through-heteroskedasticity SVAR analysis.

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

The contribution extends the observation period up to **2024**, studying the effects of uncertainty even after the **COVID-19 pandemic** and introducing additional identification restrictions.  

### **Key Findings:**  
- **Uncertainty** is more likely a **cause** of declines in economic activity.  
- **Macroeconomic regimes** should be studied separately due to **inversions** in the relationship between business cycles and uncertainty.  

### **Data Sources:**  
The data used in this project were retrieved from:  
**Jurado, Kyle, Sydney C. Ludvigson, and Serena Ng. 2015. "Measuring Uncertainty."**  
[DOI: 10.1257/aer.20131193](https://www.sydneyludvigson.com/macro-and-financial-uncertainty-indexes)  

---

## **Scripts**

### **Main Scripts:**  
- **`Structural_Var.R`**: Estimates the on-impact effects for both endogenous and endogenous model and identifies the model that fits better the data. Using the latter model plots the IRF using confidence intervals using a bootrsap approach.  
- **`Preliminary_analysis.R`**: Implements a descriptive analysis.  

### **Supporting Scripts (in `Functions/`):**  
- **`Likelihood_SVAR_Unrestricted.R`**: Performs MLE using the whole sample.  
- **`Likelihood_SVAR_Restricted_Upper.R`**: Performs MLE for the endogenous model.  
- **`Likelihood_SVAR_Restricted.R`**: Performs MLE for the exogenous model.  
- **`Likelihood_SVAR_Bootstrap.R`**: Performs MLE for the exogenous model using a bootstrapped sample.  

---

## **Instructions**

### **Loading Data:**  
- The **JLK dataset** is stored in the `data/` folder.  
- Use the script **`Preliminary_analysis.R`** to visualize a descriptive overview of the data.  
- Use the script **`Structural_Var.R`** to preprocess the data and replicate the core results.  
