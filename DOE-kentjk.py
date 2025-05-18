# Import libraries
import streamlit as st
import pandas as pd
import itertools
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.anova import anova_lm
import scipy.optimize as opt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import warnings
warnings.filterwarnings("ignore", message="kurtosistest only valid for n>=20")

# Page configurations
st.set_page_config(page_title="DOE | kentjkdigitals", layout="wide")
hide_st_style = """
                <style>
                #MainMenu {visibility:hidden;}
                footer {visibility:hidden;}
                header {visibility:hidden;}
                .block-container {padding-top: 0rem; padding-bottom: 0rem;}
                </style>
                """
st.markdown(hide_st_style, unsafe_allow_html=True)

footer = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: black;
    color: white;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 3px 10px;
    font-size: 12px !important;
    z-index: 1000;
}

.footer-left {
    text-align: left;
}

</style>

<div class="footer">
    <div class="footer-left">&copy; Kent Katigbak | Industrial Engineer | Lean Six Sigma Green Belt | Data Analyst</div>
</div>
"""

st.markdown(footer, unsafe_allow_html=True)





# App title and description
st.markdown("<h1>Design of Experiments Web App</h1>", unsafe_allow_html=True)

# Main app
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["App Introduction", "Experimental Design", "CSV File Uploader", "Graphical Analysis", "Response Optimizer", "Prediction and Simulation"])

# tab1 --- App introduction
with tab1:
    st.markdown("<h3>Welcome to the Design of Experiments Webb App</h3>", unsafe_allow_html=True)
    st.markdown("""This app is a powerful and user-friendly tool designed to streamline the planning,
            execution, and analysis of experiments. Whether you're optimizing a process, testing 
            multiple factors, or exploring relationships between variables, this app provides everything 
            you need to make data-driven decisions.""")
    st.markdown("""With features like automated design generation, statistical analysis, factorial plots, 
            and response optimization, our app empowers you to uncover significant insights quickly 
            and efficiently. Perfect for engineers, researchers, and quality professionals, the DOE 
            Web App ensures that your experiments are not only systematic but also impactful.""")
    st.markdown("""<h5>Start exploring the power of design of experiments today!</h5>""", unsafe_allow_html=True)

# tab2 --- Experimental design
with tab2:
    # Step 1: Input Factors and Levels
    st.markdown("<h5>1. Define Factors and Levels</h5>", unsafe_allow_html=True)

    # Number of factors
    num_factors = st.number_input("Enter the number of factors:", min_value=1, step=1)

    # Input factors and levels
    factors = {}
    for i in range(num_factors):
        factor_name = st.text_input(f"Name of Factor {i+1}", key=f"factor_name_{i}")
        num_levels = st.number_input(f"Number of levels for {factor_name}", min_value=2, step=1, key=f"num_levels_{i}")
        
        # Level names
        levels = []
        for j in range(num_levels):
            level = st.text_input(f"Level {j+1} for {factor_name}", key=f"level_{i}_{j}")
            levels.append(level)
        factors[factor_name] = levels

    # Number of replicates
    num_replicates = st.number_input("Enter the number of replicates:", min_value=1, step=1, value=1)

    # Generate design button
    if st.button("Generate Design"):
        # Step 2: Create the Experimental Design

        # Get all combinations of factor levels
        base_design = list(itertools.product(*factors.values()))

        # Apply replicates
        design = base_design * num_replicates

        # Create a DataFrame with an empty Response column and set the index to start at 1
        columns = list(factors.keys()) + ["Response"]
        df_design = pd.DataFrame(design, columns=factors.keys())
        df_design["Replicate"] = [i // len(base_design) + 1 for i in range(len(design))]  # Add replicate numbers
        df_design["Response"] = None  # Initialize response column with None
        
        # Set the index to start at 1 and rename it to "Run"
        df_design.index = range(1, len(df_design) + 1)
        df_design.index.name = "Run"

        # Display the design matrix
        st.markdown("<h5>Generated Design Matrix with Replicates:</h5>", unsafe_allow_html=True)
        st.dataframe(df_design)

        # Option to download the design matrix as CSV
        csv = df_design.to_csv(index=True).encode('utf-8')
        st.download_button(
            label="Download Design Matrix with Replicates as CSV",
            data=csv,
            file_name='design_matrix_with_replicates.csv',
            mime='text/csv'
        )
        
# tab3 -- File uploader
with tab3:
    
    uploaded_file = st.file_uploader("Upload your DOE data file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown("<h5>Uploaded Data:</h5>", unsafe_allow_html=True)
        st.dataframe(df)
        st.markdown("<h5>Please proceed to the nex tabs.</h5>",unsafe_allow_html=True)
    else:
        st.markdown("<h5>Please upload your experimental design csv file to proceed.</h5>",unsafe_allow_html=True)
        
# tab4 --- Graphical analysis
with tab4:
    if uploaded_file is not None:
        st.markdown("<h5>Graphical Analysis</h5>", unsafe_allow_html=True)

        response = st.selectbox("Select the Response Variable:", options=df.columns)
        factors = st.multiselect("Select Factors:", options=[col for col in df.columns if col != response])

        model_mode = st.radio("Model Type:", ["Main Effects Only", "Main + 2-Way Interactions"])

        if factors:
            for factor in factors:
                if not np.issubdtype(df[factor].dtype, np.number):
                    df[factor] = df[factor].astype('category')

            if model_mode == "Main Effects Only":
                formula = f"{response} ~ " + " + ".join([f"C({f})" for f in factors])
            else:
                main_terms = [f"C({f})" for f in factors]
                interaction_terms = [f"C({f1}):C({f2})" for i, f1 in enumerate(factors) for f2 in factors[i+1:]]
                formula = f"{response} ~ " + " + ".join(main_terms + interaction_terms)

            model = smf.ols(formula, data=df).fit()

            st.markdown("<h5>ANOVA Table (Type II)</h5>", unsafe_allow_html=True)
            st.markdown("""
            This table checks if a factor (or combination of factors) makes a real difference in the result.
            - **P-value < 0.05**: The factor likely affects the response.
            - **P-value >= 0.05**: The effect is weak or uncertain.
            Use this to filter out unimportant variables.
            """, unsafe_allow_html=True)
            anova_table = anova_lm(model, typ=2)
            st.dataframe(anova_table.round(4))

            st.markdown("<h5>Standardized Effects (Pareto Style)</h5>", unsafe_allow_html=True)
            st.markdown("""
            Bigger bars = stronger effects.
            This helps you spot what changes have the biggest effect on the results.
            - Example: A large bar for Factor A means adjusting A changes the outcome a lot.
            """, unsafe_allow_html=True)
            effects = model.params[1:]
            std_errors = model.bse[1:]
            t_vals = effects / std_errors
            effects_df = pd.DataFrame({
                "Effect": t_vals.abs().sort_values(ascending=True).index,
                "|Standardized Effect|": t_vals.abs().sort_values(ascending=True).values
            })
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(data=effects_df, y="Effect", x="|Standardized Effect|", hue="Effect", palette="Blues_d", ax=ax)
            ax.set_title("Pareto Chart of Standardized Effects")
            st.pyplot(fig, use_container_width=True)

            st.markdown("<h5>Main Effects Plot</h5>", unsafe_allow_html=True)
            st.markdown("""
            Each plot shows what happens to the average result when you change a factor.
            - If the line moves a lot: this factor matters.
            - If the line is flat: changing this factor won’t change the result much.
            """, unsafe_allow_html=True)

            for factor in factors:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.pointplot(x=factor, y=response, data=df, errorbar='sd', capsize=.1, ax=ax)
                ax.set_title(f"Main Effect of {factor}")
                ax.set_ylabel(response)
                ax.set_xlabel(factor)
                ax.tick_params(axis='x', rotation=45)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)

            st.markdown("<h5>Interaction Plots</h5>", unsafe_allow_html=True)
            st.markdown("""
            These show if two factors affect each other’s impact.
            - If the lines cross or aren’t parallel: the combination matters.
            - If the lines are mostly parallel: the effects are independent.
            """, unsafe_allow_html=True)
            for i in range(len(factors)):
                for j in range(i+1, len(factors)):
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.pointplot(x=factors[i], y=response, hue=factors[j], data=df, errorbar='sd', capsize=.1, ax=ax)
                    ax.set_title(f"Interaction: {factors[i]} x {factors[j]}")
                    # Set axis label sizes
                    ax.set_xlabel(factors[i], fontsize=10)
                    ax.set_ylabel(response, fontsize=10)
                    ax.tick_params(axis='x', rotation=45)
                    st.pyplot(fig, use_container_width=True)

            st.markdown("<h5>Residual Diagnostics (4-in-1)</h5>", unsafe_allow_html=True)
            st.markdown("""
            These 4 plots help us check if our model's predictions are reliable.

            - **Q-Q Plot**: Should form a straight line. If not, your data may not be normal.
            - **Residuals vs Fitted**: Points should be random. If there's a curve or fan-shape, the model is missing something.
            - **Histogram**: Should look like a bell shape. If skewed, the model might be off.
            - **Observation Order**: Residuals should bounce randomly around zero. A pattern means time-related issues.
            """, unsafe_allow_html=True)
            residuals = model.resid
            fitted = model.fittedvalues

            fig, axs = plt.subplots(2, 2, figsize=(8, 6))
            sm.qqplot(residuals, line='45', ax=axs[0, 0])
            axs[0, 0].set_title("Normal Q-Q")

            axs[0, 1].scatter(fitted, residuals)
            axs[0, 1].axhline(0, color='red', linestyle='--')
            axs[0, 1].set_title("Residuals vs Fitted")

            sns.histplot(residuals, kde=True, ax=axs[1, 0])
            axs[1, 0].set_title("Histogram of Residuals")

            axs[1, 1].plot(np.arange(len(residuals)), residuals, marker='o')
            axs[1, 1].axhline(0, color='red', linestyle='--')
            axs[1, 1].set_title("Residuals vs Observation Order")

            fig.tight_layout()
            st.pyplot(fig)

            st.markdown("<h5>Model Summary</h5>", unsafe_allow_html=True)
            st.markdown("""
            - **R²** tells how much of your response is explained by the factors.
            - **Adjusted R²** does the same, but adjusts for number of factors.
            - **P-values** under 0.05 suggest strong evidence the factor matters.
            - The **coefficients** tell you if the factor increases or decreases the result.
            """, unsafe_allow_html=True)
            st.text(model.summary())

            st.markdown("<h5>Breakdown of Model Summary</h5>", unsafe_allow_html=True)
            r2 = model.rsquared
            r2_adj = model.rsquared_adj
            st.markdown(f"- **R-squared (R²):** About **{r2:.2f}**, meaning the model explains about {r2*100:.1f}% of the variation in the result.")
            st.markdown(f"- **Adjusted R-squared:** About **{r2_adj:.2f}**, this accounts for how many inputs you included. Use this to compare models.")
            st.markdown("- **Note:** R² closer to 1 means better prediction. Closer to 0 means weaker model.")

            st.markdown("---")
            st.markdown("<h5>Detailed Explanation of Each Factor</h5>", unsafe_allow_html=True)
            for term, coef, pval in zip(model.params.index, model.params.values, model.pvalues.values):
                color = "green" if pval < 0.05 else "gray"
                if term == 'Intercept':
                    st.markdown(f"<span style='color:blue'>• The base response when no factors are active is **{coef:.3f}**</span>", unsafe_allow_html=True)
                else:
                    effect = "increases" if coef > 0 else "decreases"
                    signif = "<span style='color:green'>significant</span>" if pval < 0.05 else "<span style='color:gray'>not significant</span>"
                    st.markdown(f"- <span style='color:{color}'><strong>{term}</strong></span>: This factor <strong>{effect}</strong> the result by <strong>{abs(coef):.3f}</strong>. It is {signif} (p = {pval:.4f}).", unsafe_allow_html=True)

            st.markdown("<h5>Detailed Explanation of Each Factor</h5>", unsafe_allow_html=True)
            st.markdown("Below are simplified interpretations of what each model term means in your experiment.", unsafe_allow_html=True)
            for term, coef, pval in zip(model.params.index, model.params.values, model.pvalues.values):
                if term == 'Intercept':
                    st.markdown(f"- **{term}**: The baseline response when all factors are at their reference level is **{coef:.3f}**.")
                else:
                    effect = "increases" if coef > 0 else "decreases"
                    signif = "significant" if pval < 0.05 else "not statistically significant"
                    st.markdown(f"- **{term}**: This factor {effect} the response by **{abs(coef):.3f}** units on average. It is **{signif}** (p-value = {pval:.4f}).")

            st.markdown("<h5>VIF Table (Multicollinearity Check)</h5>", unsafe_allow_html=True)
            st.markdown("""
            This checks if your inputs are too similar to each other.
            - **VIF < 5**: No problem.
            - **VIF > 5**: These factors are too related; model results might be unstable.
            """, unsafe_allow_html=True)
            design_matrix = model.model.exog
            vif_data = pd.DataFrame({
                "Term": model.model.exog_names,
                "VIF": [variance_inflation_factor(design_matrix, i) for i in range(design_matrix.shape[1])]
            })
            vif_data = vif_data[vif_data['Term'] != 'Intercept']
            st.dataframe(vif_data.round(3))

            csv_export = anova_table.to_csv(index=True).encode('utf-8')
            st.download_button("Download ANOVA Table", csv_export, file_name="anova_table.csv")

        else:
            st.warning("Please select at least one factor to analyze.")
        
# tab5 --- Response Optimizer
with tab5:
    
    if uploaded_file is not None:
        
        # Step 1: Select Response Variable and Optimization Goals
        st.markdown("<h5>1. Set Response and Goals</h5>", unsafe_allow_html=True)
        
        response = st.selectbox("Select Response Variable:", options=df.columns)
        factors = st.multiselect("Select Factors for Optimization:", options=[col for col in df.columns if col != response])
        
        if factors:
            goal = st.selectbox("Goal for Response:", ["Minimize", "Maximize"])
            target = st.number_input("Target Value (Optional)", value=np.nan)
            min_bound = st.number_input("Minimum Bound", value=float(df[response].min()))
            max_bound = st.number_input("Maximum Bound", value=float(df[response].max()))

            # Step 2: Regression Model and Algebraic Formula Display
            st.markdown("<h5>2. Regression Model (Algebraic Formula)</h5>", unsafe_allow_html=True)
            
            # Create formula for regression with main effects and interactions
            formula = f"{response} ~ " + " + ".join(factors) + " + " + " + ".join([f"{i}:{j}" for i in factors for j in factors if i != j])
            model = smf.ols(formula, data=df).fit()
            
            # Extract coefficients to create an algebraic formula
            coefficients = model.params
            equation = f"{response} = "
            for i, (term, coef) in enumerate(coefficients.items()):
                if i == 0:
                    equation += f"{coef:.4f}"  # Intercept term
                else:
                    equation += f" + ({coef:.4f} * {term})"

            # Display the simplified algebraic formula
            st.latex(equation)

            # Step 3: Optimization based on Factor Levels
            st.markdown("<h5>3. Optimization</h5>", unsafe_allow_html=True)

            # Define objective function for optimization
            def objective(x):
                factor_dict = {factor: x[i] for i, factor in enumerate(factors)}
                y_pred = model.predict(pd.DataFrame([factor_dict]))[0]
                if goal == "Minimize":
                    return y_pred if np.isnan(target) else abs(y_pred - target)
                elif goal == "Maximize":
                    return -y_pred if np.isnan(target) else abs(y_pred - target)

            # Set bounds for each factor based on data range
            bounds = [(df[f].min(), df[f].max()) for f in factors]

            # Perform optimization
            initial_guess = [(df[f].mean()) for f in factors]
            result = opt.minimize(objective, initial_guess, bounds=bounds)

            if result.success:
                optimal_values = result.x
                st.write("Optimization Successful!")
                st.write("Optimal Factor Levels:")
                for factor, value in zip(factors, optimal_values):
                    st.write(f"{factor}: {value:.4f}")
                
                # Predicted response at optimal factor levels
                optimal_dict = {factor: optimal_values[i] for i, factor in enumerate(factors)}
                predicted_response = model.predict(pd.DataFrame([optimal_dict]))[0]
                st.write(f"Predicted {response} at Optimal Levels: {predicted_response:.4f}")

                # Goal achievement summary
                if not np.isnan(target):
                    if goal == "Minimize" and predicted_response <= target:
                        st.write(f"The optimization meets the target to minimize the {response} to {target}.")
                    elif goal == "Maximize" and predicted_response >= target:
                        st.write(f"The optimization meets the target to maximize the {response} to {target}.")
                    else:
                        st.write(f"The optimization does not meet the target. Predicted value is {predicted_response}, with a target of {target}.")
                else:
                    st.write(f"Optimization achieved a {goal.lower()}d {response} of {predicted_response}.")
            else:
                st.write("Optimization was unsuccessful. Try adjusting bounds or checking data.")
        else:
            st.warning("Please select at least one factor for optimization.")
    else:
        st.markdown("<h5>Please upload your experimental design csv file to proceed.</h5>",unsafe_allow_html=True)

# tab6 --- Prediction and simulation
with tab6:
    if uploaded_file is not None:
        # Allow user to input levels for each factor to predict the response
        st.markdown("<h5>Enter Factor Levels for Prediction</h5>", unsafe_allow_html=True)
        factor_levels = {}
        for factor in factors:
            level = st.number_input(f"Enter level for {factor}:", value=float(df[factor].mean()))
            factor_levels[factor] = level

        # Predict the response based on input factor levels
        if st.button("Predict Response"):
            # Convert input to DataFrame for model prediction
            input_df = pd.DataFrame([factor_levels])
            
            # Predict the response using the regression model
            predicted_response = model.predict(input_df)[0]
            
            st.subheader(f"Predicted {response}: {predicted_response:.4f}")
    else:
        st.markdown("<h5>Please upload your experimental design csv file to proceed.</h5>",unsafe_allow_html=True)

# Divider
st.divider()
