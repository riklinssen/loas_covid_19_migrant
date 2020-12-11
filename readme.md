# COVID 19- impact of Covid-19 migrant workers - LAOS

Code to reproduce analyses and visuals for Covid 19 impacts on migrant workers LAOS published here <add link> --> add link to report here. 

# Technologies
Project is created with: 
- STATA 13.1
- Python 3.8.0 

# Data
Source data for this project not public (as of Nov 2020) available on request.

# Structure
```
├───docs                    <- data documentation, questionnaire other relevant docs
│   
│          
│   
│      
│      
│      
│   
└───src                     <-.do (cleaning .py (visualisations) 
    ├───data                 
    │   ├───clean           <-Final datasets used for report generation  
    │   ├───interim         <-intermediate data that has been transformed 
    │   └───raw             <-original data dump         
    ├───graphs              <-visualisations in report (not on github)
    ├───descriptive_stats.py   <-script to generate descriptive statistics in report
    └───impact_visuals.py   <-script to generate impact/EDA statistics in report
│
└───requirements.txt   <- file that lists python packages used. Use pip install -r requirements.txt
```






