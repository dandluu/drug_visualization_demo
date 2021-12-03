# Drug Clearance, EDA, and Visiualization Challenge 

The attached data set (Lombardo 2018) contains pharmacokinetic parameters (VDss, CL) and physicochemical properties (logD, ionState) for 1352 drugs. The challenge involves wrangling and storing the data in a database and then visualizing how the pharmacokinetics depend on the drug properties.

    Column headers are located on row 9
    The data entries begin on row 10
    Relevant data for the challenge:
    Column A (“Name”) contains the drug names
    Column D (“human VDss (L/kg)”) contains the VDss values
    Column E (“human CL (mL/min/kg)”) contains the CL values
    Column R (“moka_ionState7.4”) contains ionState
    Column T (“MoKa.LogD7.4”) contains the logD values

outliers and missing values were added to the original data set

Part 1: Create a simple database to store the relevant data in the attached data set.

For instance, you could create a single MySQL table with 5 fields: drug, VDss, CL, logD, and ionState; or you can use any other database of your choosing (cloud or local, SQL or NoSQL, etc).

Part 2: Use Python to (1) query the database, (2) generate (VDss, logD) and (CL, logD) scatterplots similar to Figures 6A, 6C in the attached paper, and (3) generate two more plots that demonstrate your data processing and/or visualization skills.

For instance, you could use mysql.connector for (1), seaborn.scatterplot with points colored based on ionState for (2), and present a workflow with outlier detection and multiple seaborn plots in a Jupyter notebook for (3).

Alternative/Bonus: Create a graphical interface for the visualization or analysis.

For instance, the interface (standalone GUI or web-based) could display user selected data (e.g., histogram of CL or VDss), or it could generate a CL prediction based on a new logD value (e.g., using linear regression).

Feel free to use different approaches that demonstrate your strongest skills. The bonus is not required and can be completed in place of Part 1 or 2 if you feel that it better demonstrates your skill set.
